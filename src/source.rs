//! Unified image sources for the ingest pipeline.
//!
//! Every place Luminary pulls performer images from — pornpics, pichunter, and
//! hand-picked manual URLs — implements [`ImageSource`], so `ingest` can gather
//! from all of them in one loop and tag each stored row with its origin (the
//! `images.source` column). Adding a new source (e.g. scenes) is just one more
//! `impl ImageSource` here.
//!
//! This module also holds the two pure decision functions ingest applies to each
//! gathered image — [`classify_view`] (the front/rear/side rule) and
//! [`quality_score`] — so they can be unit-tested without the network.

use crate::pichunter::PichunterClient;
use crate::pornpics::PornpicsClient;
use async_trait::async_trait;

/// A place performer images can be gathered from. Implementations are
/// best-effort: a network or lookup failure yields an empty vec, never an error,
/// so one dead source never aborts an ingest run.
#[async_trait]
pub trait ImageSource: Send + Sync {
    /// Stable identifier written to the `images.source` column.
    fn name(&self) -> &'static str;

    /// Up to `max` *multi-angle* image URLs for `performer` — front/rear/side
    /// from within shoots, not one cover per gallery. That angle spread is what
    /// lets ingest populate every view.
    async fn gather(&self, performer: &str, max: usize) -> Vec<String>;
}

/// Splits a target image count into (galleries, per_gallery) for the gallery-
/// based sources: a handful of images from each of several shoots, so the pool
/// spans multiple sessions (varied outfits/angles) instead of over-sampling one.
fn gallery_split(max: usize) -> (usize, usize) {
    const PER_GALLERY: usize = 6;
    (max.div_ceil(PER_GALLERY).max(1), PER_GALLERY)
}

#[async_trait]
impl ImageSource for PornpicsClient {
    fn name(&self) -> &'static str {
        "pornpics"
    }

    async fn gather(&self, performer: &str, max: usize) -> Vec<String> {
        let (galleries, per) = gallery_split(max);
        let mut urls = self.gallery_image_urls(performer, galleries, per).await;
        urls.truncate(max);
        urls
    }
}

#[async_trait]
impl ImageSource for PichunterClient {
    fn name(&self) -> &'static str {
        "pichunter"
    }

    async fn gather(&self, performer: &str, max: usize) -> Vec<String> {
        let (galleries, per) = gallery_split(max);
        let mut urls = self.gallery_image_urls(performer, galleries, per).await;
        urls.truncate(max);
        urls
    }
}

/// A fixed list of hand-picked URLs — the sanctioned route for specific images
/// the user has vetted. Ignores `performer`/`max`: these were chosen by hand.
pub struct ManualSource {
    pub urls: Vec<String>,
}

#[async_trait]
impl ImageSource for ManualSource {
    fn name(&self) -> &'static str {
        "manual"
    }

    async fn gather(&self, _performer: &str, _max: usize) -> Vec<String> {
        self.urls.clone()
    }
}

/// Resolves an image's view from the pose classifier's coarse output and whether
/// a face was detected.
///
/// THE rule (see `body_embed.py::classify_view`): pose alone can only tell
/// `side` from `frontal`; front-vs-rear comes from the *face*. MediaPipe reports
/// face-landmark visibility as high even when the subject is turned away, so a
/// "frontal" body *with* a detected face is a front shot, one *without* is a rear
/// shot. With no usable pose, a detected face is still a frontal head/torso crop;
/// nothing at all is genuinely `unknown`.
pub fn classify_view(pose_view: Option<&str>, has_face: bool) -> &'static str {
    match (pose_view, has_face) {
        (Some("side"), _) => "side",
        (Some("frontal"), true) => "front",
        (Some("frontal"), false) => "rear",
        (_, true) => "front",
        (_, false) => "unknown",
    }
}

/// A 0–1 quality/confidence weight for an ingested image, used later to weight
/// its contribution to the per-view body centroids. Blends identity confidence
/// (how well the face matches the seed) with body completeness (did the pose and
/// seg vectors clear their full-body gates).
///
/// `id_sim` is the cosine similarity of this image's face to the performer's seed
/// face, or `None` when there's no face to compare (a rear shot, or no seed at
/// all) — identity is then gallery-trusted at a neutral 0.5 rather than confirmed.
pub fn quality_score(id_sim: Option<f32>, has_pose: bool, has_seg: bool) -> f32 {
    let id = match id_sim {
        // Saturate at 0.8: ArcFace same-identity cosine rarely exceeds it, so we
        // don't demand a near-duplicate to count identity as fully confirmed.
        Some(s) => (s.max(0.0) / 0.8).min(1.0),
        None => 0.5,
    };
    let body = 0.5 * has_pose as u8 as f32 + 0.5 * has_seg as u8 as f32;
    0.5 * id + 0.5 * body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_front_rear_comes_from_face_not_pose() {
        // A frontal-axis body splits on the face: present => front, absent => rear.
        assert_eq!(classify_view(Some("frontal"), true), "front");
        assert_eq!(classify_view(Some("frontal"), false), "rear");
        // Side is side regardless of whether a (turned) face was seen.
        assert_eq!(classify_view(Some("side"), true), "side");
        assert_eq!(classify_view(Some("side"), false), "side");
        // No pose: a face is a frontal crop; nothing at all is unknown.
        assert_eq!(classify_view(None, true), "front");
        assert_eq!(classify_view(None, false), "unknown");
    }

    #[test]
    fn quality_rewards_identity_and_body_completeness() {
        let best = quality_score(Some(0.8), true, true); // confirmed, full body
        let portrait = quality_score(Some(0.8), false, false); // confirmed, face only
        let rear = quality_score(None, true, true); // unverifiable, full body
        assert!((best - 1.0).abs() < 1e-6);
        assert!((portrait - 0.5).abs() < 1e-6);
        assert!((rear - 0.75).abs() < 1e-6);
        // Gold-standard front beats a rear shot beats a bodiless portrait.
        assert!(best > rear && rear > portrait);
    }

    #[test]
    fn quality_floors_identity_at_zero_for_negative_sim() {
        // A rejected-range negative similarity can't drag quality below the
        // body contribution.
        assert_eq!(quality_score(Some(-0.5), false, false), 0.0);
    }

    #[test]
    fn gallery_split_spreads_across_shoots() {
        assert_eq!(gallery_split(24), (4, 6));
        assert_eq!(gallery_split(1), (1, 6)); // always at least one gallery
    }

    #[tokio::test]
    async fn manual_source_returns_its_urls_verbatim() {
        let src = ManualSource {
            urls: vec!["https://example/a.jpg".to_string()],
        };
        assert_eq!(src.name(), "manual");
        assert_eq!(
            src.gather("ignored", 99).await,
            vec!["https://example/a.jpg".to_string()]
        );
    }
}
