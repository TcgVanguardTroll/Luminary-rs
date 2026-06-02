//! Integration tests exercising the public `luminary` library API end-to-end
//! (no network, no DB) — the kind of test the lib/binary split enables.

use luminary::config::{Config, GenderFilter};
use luminary::models::Performer;
use luminary::recommender;

fn perf(name: &str, body: &str, eth: &str, hair: &str, age: u32, meas: &str) -> Performer {
    let mut p = Performer::new(name.to_string());
    p.body_type = body.to_string();
    p.ethnicity = Some(eth.to_string());
    p.hair_color = Some(hair.to_string());
    p.age = Some(age);
    p.measurements = Some(meas.to_string());
    p
}

#[test]
fn profile_tree_and_dominant_path() {
    let people = vec![
        perf("a", "Curvy", "Caucasian", "Blonde", 50, "36DD-27-38"),
        perf("b", "Curvy", "Caucasian", "Blonde", 48, "34DD-26-36"),
        perf("c", "Curvy", "Caucasian", "Brunette", 40, "34C-25-35"),
        perf("d", "Slim", "Asian", "Black", 24, "32A-24-34"),
    ];
    let tree = recommender::build_preference_tree(&people);
    let path = recommender::dominant_query_path(&tree);

    // Curvy (3/4) is the dominant body type, then Caucasian, then Blonde.
    assert_eq!(path.first().map(String::as_str), Some("Curvy"));
    assert_eq!(path.get(1).map(String::as_str), Some("Caucasian"));
    assert_eq!(path.get(2).map(String::as_str), Some("Blonde"));
}

#[test]
fn idf_then_score_prefers_distinctive_match() {
    let people = vec![
        perf("a", "Curvy", "Caucasian", "Blonde", 50, "36DD-27-38"),
        perf("b", "Curvy", "Caucasian", "Blonde", 48, "34DD-26-36"),
        perf("c", "Curvy", "Latin", "Brunette", 30, "34C-25-35"),
    ];
    let tree = recommender::build_preference_tree(&people);
    let idf = recommender::compute_idf_weights(&people);

    let on_type = perf("x", "Curvy", "Caucasian", "Blonde", 50, "36DD-27-38");
    let off_type = perf("y", "Slim", "Caucasian", "Blonde", 50, "32A-24-34");

    // Body type is a hard gate: a Slim candidate scores zero against a Curvy tree.
    assert!(recommender::score_performer_idf(&on_type, &tree, &idf) > 0.0);
    assert_eq!(
        recommender::score_performer_idf(&off_type, &tree, &idf),
        0.0
    );
}

#[test]
fn gender_filter_default_excludes_trans() {
    let cfg = Config::default();
    assert_eq!(cfg.gender_filter, GenderFilter::Female);
    assert!(cfg.gender_filter.matches(Some("Female")));
    assert!(!cfg.gender_filter.matches(Some("Transgender Female")));
}

#[test]
fn whr_drives_build_similarity() {
    // Dee-Siren-like hourglass vs an average build, compared to a twin.
    let reference = perf("ref", "Curvy", "Caucasian", "Blonde", 40, "34B-24-36"); // whr .667
    let twin = perf("twin", "Curvy", "Caucasian", "Blonde", 41, "34B-24-36");
    let average = perf("avg", "Curvy", "Caucasian", "Blonde", 40, "34B-28-34"); // whr .82

    let rv = recommender::feature_vector(&reference).unwrap();
    let tv = recommender::feature_vector(&twin).unwrap();
    let av = recommender::feature_vector(&average).unwrap();
    assert!(rv.similarity_pct(&tv) > rv.similarity_pct(&av));
}
