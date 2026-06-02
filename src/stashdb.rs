//! Minimal StashDB (stash-box) GraphQL client used to enrich a performer with
//! additional face images. StashDB exposes an `images` array per performer,
//! which gives the centroid face embedding several real photos to average —
//! noticeably more robust than a single TPDB face crop.
//!
//! Used purely for image enrichment; TPDB remains the primary metadata source.

use anyhow::{Context, Result};
use serde::Deserialize;

const STASHDB_GRAPHQL: &str = "https://stashdb.org/graphql";

pub struct StashdbClient {
    client: reqwest::Client,
    api_key: String,
}

#[derive(Deserialize)]
struct GqlResponse {
    data: Option<SearchData>,
}

#[derive(Deserialize)]
struct SearchData {
    #[serde(rename = "searchPerformers")]
    search_performers: SearchResult,
}

#[derive(Deserialize)]
struct SearchResult {
    performers: Vec<StashPerformer>,
}

#[derive(Deserialize)]
struct StashPerformer {
    #[serde(default)]
    images: Vec<StashImage>,
}

#[derive(Deserialize)]
struct StashImage {
    url: String,
    #[serde(default)]
    width: u32,
    #[serde(default)]
    height: u32,
}

impl StashdbClient {
    pub fn new(api_key: String) -> Self {
        let client = reqwest::Client::builder()
            .user_agent("Luminary/0.1.0")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();
        StashdbClient { client, api_key }
    }

    /// Returns up to `max` image URLs for the best name match, largest first.
    /// Best-effort: returns an empty vec on any error so callers can fall back.
    pub async fn image_urls(&self, name: &str, max: usize) -> Vec<String> {
        match self.try_image_urls(name, max).await {
            Ok(urls) => urls,
            Err(e) => {
                log::warn!("StashDB lookup failed for {}: {}", name, e);
                Vec::new()
            }
        }
    }

    async fn try_image_urls(&self, name: &str, max: usize) -> Result<Vec<String>> {
        let query = r#"
            query($term: String!) {
                searchPerformers(term: $term, limit: 1) {
                    performers { images { url width height } }
                }
            }
        "#;
        let body = serde_json::json!({
            "query": query,
            "variables": { "term": name },
        });

        let resp = self
            .client
            .post(STASHDB_GRAPHQL)
            .header("ApiKey", &self.api_key)
            .json(&body)
            .send()
            .await
            .context("StashDB request failed")?;

        if !resp.status().is_success() {
            anyhow::bail!("StashDB returned {}", resp.status());
        }

        let parsed: GqlResponse = resp.json().await.context("parse StashDB response")?;
        let performers = parsed
            .data
            .map(|d| d.search_performers.performers)
            .unwrap_or_default();

        let Some(p) = performers.into_iter().next() else {
            return Ok(Vec::new());
        };

        // Largest images first — better for face detection / embedding.
        let mut imgs = p.images;
        imgs.sort_by_key(|i| std::cmp::Reverse(i.width as u64 * i.height as u64));
        Ok(imgs.into_iter().take(max).map(|i| i.url).collect())
    }
}
