//! Plain-English query parsing → the structured inputs `find` already takes.
//!
//! Local and rule-based (no LLM, keeping the engine privacy-first): keyword-spot
//! the colouring/ethnicity attributes, and split `"<trait> like <name>"` clauses
//! into face references ("looks like X") vs body references ("butt like Y",
//! "built like Z"). Handles the structured shapes — e.g. "blue-eyed blondes that
//! look like Naughty Alysha", "blue-eyed blondes with a butt like Dee Siren",
//! and the two-reference form "… look like X with a butt like Y". Names are
//! resolved against the library later by `find`.

/// The structured pieces parsed out of a plain-English query — a subset of
/// `find`'s flags (the ones natural language commonly expresses).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ParsedQuery {
    /// Face references ("looks like X").
    pub looks_like: Vec<String>,
    /// Body/butt references ("butt like Y", "built like Z").
    pub body_like: Vec<String>,
    pub hair: Option<String>,
    pub eye: Option<String>,
    pub ethnicity: Option<String>,
}

// Filler words skipped when finding a clause's qualifier, and dropped from the
// attribute head. `that`/`with`/`and` also break a reference name.
const STOP: &[&str] = &[
    "a", "an", "the", "with", "of", "her", "his", "is", "are", "that",
];
const FACE_QUAL: &[&str] = &[
    "look",
    "looks",
    "looking",
    "face",
    "resembles",
    "resembling",
];
const BODY_QUAL: &[&str] = &[
    "butt", "ass", "booty", "rear", "body", "build", "built", "figure", "frame", "shape",
];
const BREAKS: &[&str] = &["with", "and", "that", "who", "but", "plus"];

const EYE: &[(&str, &str)] = &[
    ("blue", "Blue"),
    ("green", "Green"),
    ("hazel", "Hazel"),
    ("grey", "Grey"),
    ("gray", "Grey"),
    ("brown eye", "Brown"),
];
const HAIR: &[(&str, &str)] = &[
    ("blonde", "Blonde"),
    ("blond", "Blonde"),
    ("brunette", "Brunette"),
    ("redhead", "Red"),
    ("red hair", "Red"),
    ("black hair", "Black"),
    ("auburn", "Auburn"),
];
const ETHNICITY: &[(&str, &str)] = &[
    ("caucasian", "Caucasian"),
    ("latina", "Latin"),
    ("latin", "Latin"),
    ("ebony", "Black"),
    ("asian", "Asian"),
    ("indian", "Indian"),
];

/// Parses a free-text query into [`ParsedQuery`]. Pure (no DB); name resolution
/// happens downstream in `find`.
pub fn parse(text: &str) -> ParsedQuery {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    let mut q = ParsedQuery::default();

    // ── Reference clauses: each "like <name>" preceded by a trait qualifier ──
    let mut i = 0;
    while i + 1 < words.len() {
        if words[i] != "like" {
            i += 1;
            continue;
        }
        // The qualifier is the nearest non-filler word before "like".
        let qual = words[..i]
            .iter()
            .rev()
            .find(|w| !STOP.contains(w))
            .copied()
            .unwrap_or("look");
        let is_body = BODY_QUAL.contains(&qual);
        // The name runs from after "like" to the next break word (or the end).
        let start = i + 1;
        let mut j = start;
        while j < words.len() && !BREAKS.contains(&words[j]) {
            j += 1;
        }
        let name = titlecase(&words[start..j].join(" "));
        if !name.is_empty() {
            if is_body {
                q.body_like.push(name);
            } else {
                q.looks_like.push(name);
            }
        }
        i = j;
    }

    // ── Attribute head: everything before the first clause's qualifier ──
    let head_words: &[&str] = match words.iter().position(|w| *w == "like") {
        Some(li) => {
            let mut end = li;
            while end > 0
                && (STOP.contains(&words[end - 1])
                    || FACE_QUAL.contains(&words[end - 1])
                    || BODY_QUAL.contains(&words[end - 1]))
            {
                end -= 1;
            }
            &words[..end]
        }
        None => &words[..],
    };
    let head = head_words.join(" ");
    q.eye = pick(&head, EYE);
    q.hair = pick(&head, HAIR);
    q.ethnicity = pick(&head, ETHNICITY);
    q
}

/// First table entry whose keyword appears in `head`.
fn pick(head: &str, table: &[(&str, &str)]) -> Option<String> {
    table
        .iter()
        .find(|(kw, _)| head.contains(kw))
        .map(|(_, v)| v.to_string())
}

/// Title-cases each word of a name ("naughty alysha" -> "Naughty Alysha").
fn titlecase(s: &str) -> String {
    s.split_whitespace()
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eyed_blondes_who_look_like() {
        let q = parse("blue eyed blondes that look like Naughty Alysha");
        assert_eq!(q.eye.as_deref(), Some("Blue"));
        assert_eq!(q.hair.as_deref(), Some("Blonde"));
        assert_eq!(q.looks_like, vec!["Naughty Alysha".to_string()]);
        assert!(q.body_like.is_empty());
    }

    #[test]
    fn eyed_blondes_with_a_butt_like() {
        let q = parse("blue eyed blondes with a butt like Dee Siren");
        assert_eq!(q.eye.as_deref(), Some("Blue"));
        assert_eq!(q.hair.as_deref(), Some("Blonde"));
        assert_eq!(q.body_like, vec!["Dee Siren".to_string()]);
        assert!(q.looks_like.is_empty());
    }

    #[test]
    fn two_references_face_and_butt() {
        let q = parse("brunettes that look like Riley Reid with a butt like Jada Stevens");
        assert_eq!(q.hair.as_deref(), Some("Brunette"));
        assert_eq!(q.looks_like, vec!["Riley Reid".to_string()]);
        assert_eq!(q.body_like, vec!["Jada Stevens".to_string()]);
    }

    #[test]
    fn brown_eyes_need_the_eye_cue() {
        // "brown" alone (hair context) must not be read as brown eyes.
        let q = parse("brown eyed latina built like Sheena Ryder");
        assert_eq!(q.eye.as_deref(), Some("Brown"));
        assert_eq!(q.ethnicity.as_deref(), Some("Latin"));
        assert_eq!(q.body_like, vec!["Sheena Ryder".to_string()]);
    }
}
