use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use crate::models::Performer;

const TPDB_API_BASE: &str = "https://api.theporndb.net";

/// ThePornDB API client
pub struct TpdbClient {
    client: reqwest::Client,
    api_key: String,
}

#[derive(Debug, Deserialize)]
struct TpdbSearchResponse {
    data: Vec<TpdbPerformer>,
}

#[derive(Debug, Deserialize)]
struct TpdbPerformerResponse {
    data: TpdbPerformer,
}

#[derive(Debug, Deserialize, Serialize)]
struct TpdbPerformer {
    id: String,
    name: String,
    #[serde(default)]
    extras: TpdbExtras,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    images: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct TpdbExtras {
    #[serde(default)]
    birthday: Option<String>,
    #[serde(default)]
    age: Option<u32>,
    #[serde(default)]
    ethnicity: Option<String>,
    #[serde(default)]
    hair_color: Option<String>,
    #[serde(default)]
    eye_color: Option<String>,
    #[serde(default)]
    height: Option<String>,
    #[serde(default)]
    weight: Option<String>,
    #[serde(default)]
    measurements: Option<String>,
}

impl TpdbClient {
    pub fn new(api_key: String) -> Self {
        let client = reqwest::Client::builder()
            .user_agent("Starfinder/0.1.0")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();

        TpdbClient { client, api_key }
    }

    /// Search for a performer by name
    pub async fn search_performer(&self, name: &str) -> Result<Option<Performer>> {
        let url = format!("{}/performers?q={}", TPDB_API_BASE, urlencoding::encode(name));

        log::info!("Searching ThePornDB: {}", name);

        let response = self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .context("Failed to search ThePornDB")?;

        if !response.status().is_success() {
            anyhow::bail!("ThePornDB returned status: {}", response.status());
        }

        let search_result: TpdbSearchResponse = response.json().await
            .context("Failed to parse ThePornDB search response")?;

        if let Some(tpdb_performer) = search_result.data.first() {
            self.get_performer(&tpdb_performer.id).await
        } else {
            Ok(None)
        }
    }

    /// Get detailed performer information by ID
    pub async fn get_performer(&self, id: &str) -> Result<Option<Performer>> {
        let url = format!("{}/performers/{}", TPDB_API_BASE, id);

        log::info!("Fetching performer details: {}", id);

        let response = self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .context("Failed to fetch performer from ThePornDB")?;

        if !response.status().is_success() {
            anyhow::bail!("ThePornDB returned status: {}", response.status());
        }

        let performer_response: TpdbPerformerResponse = response.json().await
            .context("Failed to parse ThePornDB performer response")?;

        Ok(Some(self.convert_to_performer(performer_response.data)))
    }

    /// Convert ThePornDB performer to our Performer model
    fn convert_to_performer(&self, tpdb: TpdbPerformer) -> Performer {
        let mut performer = Performer::new(tpdb.name.clone());

        performer.body_type = self.infer_body_type(&tpdb);

        performer.age = tpdb.extras.age;
        performer.ethnicity = tpdb.extras.ethnicity;
        performer.hair_color = tpdb.extras.hair_color;
        performer.eye_color = tpdb.extras.eye_color;
        performer.height = tpdb.extras.height;
        performer.weight = tpdb.extras.weight;
        performer.measurements = tpdb.extras.measurements;
        performer.birthdate = tpdb.extras.birthday;

        performer.profile_image_url = tpdb.image.or_else(|| tpdb.images.first().cloned());
        performer.gallery_urls = tpdb.images;

        performer.source = Some("ThePornDB".to_string());
        performer.source_url = Some(format!("https://theporndb.net/performers/{}", tpdb.id));
        performer.last_updated = Some(chrono::Utc::now().to_rfc3339());

        performer
    }

    /// Infer body type from available data
    fn infer_body_type(&self, tpdb: &TpdbPerformer) -> String {
        if let Some(ref measurements) = tpdb.extras.measurements {
            if measurements.contains("DD") || measurements.contains("E") || measurements.contains("F") {
                return "Curvy".to_string();
            }
        }
        "Average".to_string()
    }
}
