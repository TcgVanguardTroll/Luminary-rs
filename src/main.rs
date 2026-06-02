use clap::{Parser, Subcommand};
use colored::*;
use anyhow::Context;

mod models;
mod database;
mod image_cache;
mod scraper;
mod tpdb;

use database::Database;
use models::SearchFilters;
use scraper::Scraper;
use tpdb::TpdbClient;

#[derive(Parser)]
#[command(name = "starfinder")]
#[command(about = "Find your stars - A Rust-powered recommendation engine", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Add performers you like to build your profile
    Add {
        /// Names of performers to add
        #[arg(required = true)]
        names: Vec<String>,
    },
    /// List all performers in your database
    List,
    /// Search for performers by attributes
    Search {
        #[arg(long)]
        body_type: Option<String>,
        #[arg(long)]
        age_min: Option<u32>,
        #[arg(long)]
        age_max: Option<u32>,
        #[arg(long)]
        ethnicity: Option<String>,
        #[arg(long)]
        hair_color: Option<String>,
        #[arg(long, default_value_t = false)]
        show_images: bool,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// View details of a specific performer
    View {
        name: String,
        #[arg(long, default_value_t = false)]
        gallery: bool,
    },
    /// Remove a performer from the database
    Remove {
        name: String,
    },
    /// Show statistics about your database
    Stats,
    /// Clear image cache
    ClearCache,
    /// Import performers from JSON file
    Import {
        #[arg(default_value = "performers_data.json")]
        file: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();
    let db_path = get_db_path()?;
    let db = Database::new(&db_path)?;

    match cli.command {
        Commands::Add { names } => {
            add_performers(&db, names).await?;
        }
        Commands::List => {
            list_performers(&db)?;
        }
        Commands::Search {
            body_type,
            age_min,
            age_max,
            ethnicity,
            hair_color,
            show_images,
            limit,
        } => {
            search_performers(
                &db, body_type, age_min, age_max,
                ethnicity, hair_color, show_images, limit,
            )
            .await?;
        }
        Commands::View { name, gallery } => {
            view_performer(&db, &name, gallery).await?;
        }
        Commands::Remove { name } => {
            remove_performer(&db, &name)?;
        }
        Commands::Stats => {
            show_stats(&db)?;
        }
        Commands::ClearCache => {
            clear_cache()?;
        }
        Commands::Import { file } => {
            import_from_json(&db, &file)?;
        }
    }

    Ok(())
}

fn get_db_path() -> anyhow::Result<String> {
    let data_dir = dirs::data_local_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not find data directory"))?;
    let db_dir = data_dir.join("starfinder");
    std::fs::create_dir_all(&db_dir)?;
    Ok(db_dir.join("starfinder.db").to_string_lossy().to_string())
}

async fn add_performers(db: &Database, names: Vec<String>) -> anyhow::Result<()> {
    println!("{}", "Adding performers to your profile...".bright_cyan().bold());
    println!();

    let api_key = std::env::var("TPDB_API_KEY").ok();
    let tpdb_client = api_key.as_ref().map(|key| TpdbClient::new(key.clone()));
    let scraper = Scraper::new();

    if tpdb_client.is_some() {
        println!("{}", "Using ThePornDB API".bright_green());
    } else {
        println!("{}", "No TPDB_API_KEY found, using web scraping (may fail)".yellow());
        println!("{}", "   Set TPDB_API_KEY environment variable for better results".bright_black());
    }
    println!();

    for name in names {
        print!("{} ", format!("Fetching data for {}...", name).bright_black());
        std::io::Write::flush(&mut std::io::stdout()).ok();

        let mut performer_data = None;

        if let Some(ref client) = tpdb_client {
            match client.search_performer(&name).await {
                Ok(Some(performer)) => {
                    performer_data = Some(performer);
                }
                Ok(None) => {
                    log::warn!("ThePornDB: No results for {}", name);
                }
                Err(e) => {
                    log::warn!("ThePornDB error for {}: {}", name, e);
                }
            }
        }

        if performer_data.is_none() {
            match scraper.scrape_performer(&name).await {
                Ok(performer) => {
                    performer_data = Some(performer);
                }
                Err(e) => {
                    log::warn!("Scraper error for {}: {}", name, e);
                }
            }
        }

        match performer_data {
            Some(performer) => {
                let source = performer.source.as_deref().unwrap_or("Unknown");
                match db.add_performer(&performer) {
                    Ok(_) => {
                        println!("\r{} {} {} {}",
                            "[OK]".green(),
                            "Added:".white(),
                            name.bright_white().bold(),
                            format!("({}, {})", performer.body_type, source).bright_black()
                        );
                    }
                    Err(e) => {
                        println!("\r{} {} {}: {}",
                            "[X]".red(), "Failed to save:".white(), name, e);
                    }
                }
            }
            None => {
                println!("\r{} {} {}",
                    "[X]".red(), "Not found:".white(), name);
            }
        }
    }

    println!();
    println!("{}", "Tip: Use 'starfinder search' to find similar performers".bright_black());
    Ok(())
}

fn list_performers(db: &Database) -> anyhow::Result<()> {
    let performers = db.get_all_performers()?;

    if performers.is_empty() {
        println!("{}", "No performers in database yet.".yellow());
        println!("{}", "Use 'starfinder add <names...>' to add some!".bright_black());
        return Ok(());
    }

    println!("{}", format!("Your Performers ({} total)", performers.len()).bright_cyan().bold());
    println!();

    for performer in performers {
        println!("{} {} {}",
            "*".bright_black(),
            performer.name.bright_white().bold(),
            format!("({})", performer.body_type).bright_black()
        );
    }

    Ok(())
}

async fn search_performers(
    db: &Database,
    body_type: Option<String>,
    age_min: Option<u32>,
    age_max: Option<u32>,
    ethnicity: Option<String>,
    hair_color: Option<String>,
    _show_images: bool,
    limit: usize,
) -> anyhow::Result<()> {
    let filters = SearchFilters {
        body_type, age_min, age_max, ethnicity, hair_color,
        categories: Vec::new(),
        min_score: None,
    };

    let results = db.search(&filters)?;

    if results.is_empty() {
        println!("{}", "No matches found.".yellow());
        return Ok(());
    }

    println!("{}", format!("Search Results ({} matches)", results.len()).bright_cyan().bold());
    println!();

    for (i, performer) in results.iter().take(limit).enumerate() {
        println!("{}. {} {}",
            (i + 1).to_string().bright_black(),
            performer.name.bright_white().bold(),
            format!("(Age: {}, Body: {})",
                performer.age.map(|a| a.to_string()).unwrap_or_else(|| "?".to_string()),
                performer.body_type
            ).bright_black()
        );
    }

    Ok(())
}

async fn view_performer(db: &Database, name: &str, _gallery: bool) -> anyhow::Result<()> {
    match db.get_performer(name)? {
        Some(performer) => {
            println!("{}", "=".repeat(50).bright_black());
            println!("{}", performer.name.bright_white().bold());
            println!("{}", "=".repeat(50).bright_black());
            println!();

            println!("  {} {}", "Body Type:".bright_black(), performer.body_type);
            if let Some(age) = performer.age {
                println!("  {} {}", "Age:".bright_black(), age);
            }
            if let Some(eth) = performer.ethnicity {
                println!("  {} {}", "Ethnicity:".bright_black(), eth);
            }
            if let Some(hair) = performer.hair_color {
                println!("  {} {}", "Hair:".bright_black(), hair);
            }
            if let Some(meas) = performer.measurements {
                println!("  {} {}", "Measurements:".bright_black(), meas);
            }
            println!();
        }
        None => {
            println!("{}", format!("Performer '{}' not found in database.", name).red());
        }
    }
    Ok(())
}

fn remove_performer(db: &Database, name: &str) -> anyhow::Result<()> {
    db.remove_performer(name)?;
    println!("{} {}", "Removed:".green(), name);
    Ok(())
}

fn show_stats(db: &Database) -> anyhow::Result<()> {
    let count = db.count()?;
    let cache = image_cache::ImageCache::new()?;
    let cache_size = cache.cache_size()?;
    let cache_count = cache.cache_count()?;

    println!("{}", "Starfinder Statistics".bright_cyan().bold());
    println!();
    println!("  {} {}", "Performers:".bright_black(), count.to_string().bright_white());
    println!("  {} {}", "Cached Images:".bright_black(), cache_count.to_string().bright_white());
    println!("  {} {:.2} MB", "Cache Size:".bright_black(),
        cache_size as f64 / 1024.0 / 1024.0);
    println!();

    Ok(())
}

fn clear_cache() -> anyhow::Result<()> {
    let cache = image_cache::ImageCache::new()?;
    cache.clear()?;
    println!("{}", "Image cache cleared".green());
    Ok(())
}

fn import_from_json(db: &Database, file_path: &str) -> anyhow::Result<()> {
    use std::fs;
    use models::Performer;

    println!("{}", format!("Importing performers from {}...", file_path).bright_cyan().bold());
    println!();

    let json_data = fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read {}", file_path))?;

    let performers: Vec<Performer> = serde_json::from_str(&json_data)
        .with_context(|| format!("Failed to parse JSON from {}", file_path))?;

    let mut success_count = 0;
    let mut fail_count = 0;

    for performer in performers {
        match db.add_performer(&performer) {
            Ok(_) => {
                println!("{} {} {}",
                    "[OK]".green(), "Imported:".white(),
                    performer.name.bright_white().bold());
                success_count += 1;
            }
            Err(e) => {
                println!("{} {} {}: {}",
                    "[X]".red(), "Failed:".white(), performer.name, e);
                fail_count += 1;
            }
        }
    }

    println!();
    println!("{}", format!("Imported {} performers ({} failed)", success_count, fail_count).green());
    Ok(())
}
