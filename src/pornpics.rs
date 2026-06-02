//! Minimal pornpics.com image source.
//!
//! Used purely as an extra pool of *full-body* images for building body-frame
//! centroids — niche performers often have only one or two profile photos
//! elsewhere, and a pornstar gallery page yields ~20 distinct full-body shoots.
//!
//! Scope/ethics: pornpics.com's robots.txt permits `/pornstars/` and sets no
//! AI/crawler restriction (unlike IAFD, which we deliberately do not touch).
//! We fetch one public gallery-index page and read the image URLs already in it
//! — for personal, local model-building only, never redistributed.

use anyhow::{Context, Result};

const BASE: &str = "https://www.pornpics.com";
/// Gallery thumbnails are served at this width; we swap it for a larger one so
/// pose estimation has enough pixels to work with.
const THUMB_SEG: &str = "/460/";
const FULL_SEG: &str = "/1280/";

pub struct PornpicsClient {
    client: reqwest::Client,
}

impl Default for PornpicsClient {
    fn default() -> Self {
        Self::new()
    }
}

impl PornpicsClient {
    pub fn new() -> Self {
        // A browser User-Agent — the site serves an empty page to non-browser
        // agents.
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();
        PornpicsClient { client }
    }

    /// Returns up to `max` full-body image URLs for a performer from their
    /// pornpics.com pornstar page. Best-effort: returns an empty vec on any
    /// error (no such page, network failure) so callers can fall back.
    pub async fn image_urls(&self, name: &str, max: usize) -> Vec<String> {
        match self.try_image_urls(name, max).await {
            Ok(urls) => urls,
            Err(e) => {
                log::warn!("pornpics lookup failed for {}: {}", name, e);
                Vec::new()
            }
        }
    }

    async fn try_image_urls(&self, name: &str, max: usize) -> Result<Vec<String>> {
        let url = format!("{}/pornstars/{}/", BASE, slugify(name));
        log::info!("pornpics: {}", url);

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("pornpics request failed")?;
        if !resp.status().is_success() {
            anyhow::bail!("pornpics returned {}", resp.status());
        }
        let body = resp.text().await.context("read pornpics body")?;
        Ok(extract_image_urls(&body, max))
    }
}

/// "Christina Sapphire" -> "christina-sapphire" (lowercase, runs of non-alnum
/// collapsed to a single hyphen, trimmed).
fn slugify(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_dash = false;
    for ch in name.trim().to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// Scans raw HTML for `cdni.pornpics.com<THUMB_SEG>...jpg` gallery thumbnails
/// and upscales each to a pose-usable width by swapping the size segment.
/// Deduplicated, preserving page order, capped at `max`.
fn extract_image_urls(html: &str, max: usize) -> Vec<String> {
    let marker = format!("https://cdni.pornpics.com{}", THUMB_SEG);
    let mut urls: Vec<String> = Vec::new();
    let mut rest = html;

    while let Some(pos) = rest.find(&marker) {
        let tail = &rest[pos..];
        // The URL ends at the first delimiter (quote/space/escape/paren).
        let end = tail
            .find(|c: char| c == '"' || c == '\'' || c == '\\' || c == ')' || c.is_whitespace())
            .unwrap_or(tail.len());
        let raw = &tail[..end];

        if raw.ends_with(".jpg") || raw.ends_with(".jpeg") || raw.ends_with(".webp") {
            let big = raw.replacen(THUMB_SEG, FULL_SEG, 1);
            if !urls.contains(&big) {
                urls.push(big);
                if urls.len() >= max {
                    break;
                }
            }
        }
        rest = &tail[end..];
    }
    urls
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_handles_spaces_and_punctuation() {
        assert_eq!(slugify("Christina Sapphire"), "christina-sapphire");
        assert_eq!(slugify("  Anna  Claire   Clouds "), "anna-claire-clouds");
        assert_eq!(slugify("Vicky O'Hara"), "vicky-o-hara");
    }

    #[test]
    fn extract_upscales_and_dedupes_in_order() {
        let html = r#"
            <img src="https://cdni.pornpics.com/460/7/445/111/111_001_a.jpg">
            <img data-src="https://cdni.pornpics.com/460/7/445/222/222_002_b.jpg"/>
            <img src="https://cdni.pornpics.com/460/7/445/111/111_001_a.jpg">
        "#;
        let urls = extract_image_urls(html, 10);
        assert_eq!(
            urls,
            vec![
                "https://cdni.pornpics.com/1280/7/445/111/111_001_a.jpg".to_string(),
                "https://cdni.pornpics.com/1280/7/445/222/222_002_b.jpg".to_string(),
            ]
        );
    }

    #[test]
    fn extract_respects_max() {
        let html = "a https://cdni.pornpics.com/460/1/1/1/1_1_a.jpg \
                    b https://cdni.pornpics.com/460/1/1/2/2_2_b.jpg \
                    c https://cdni.pornpics.com/460/1/1/3/3_3_c.jpg";
        assert_eq!(extract_image_urls(html, 2).len(), 2);
    }

    #[test]
    fn extract_ignores_non_images() {
        let html = "https://cdni.pornpics.com/460/1/1/1/page.html and text";
        assert!(extract_image_urls(html, 10).is_empty());
    }
}
