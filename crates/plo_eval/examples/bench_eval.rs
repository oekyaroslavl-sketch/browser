use std::time::Instant;
use plo_cards::{Card, Hand4};
use plo_eval::{eval_plo_river, init_tables, RiverBoardCache};

fn main() {
    init_tables();

    // Generate 5,000 live 4-card PLO hands on a fixed river board [As Ks Td 7h 2c]
    let river = [
        "As".parse::<Card>().unwrap(),
        "Ks".parse::<Card>().unwrap(),
        "Td".parse::<Card>().unwrap(),
        "7h".parse::<Card>().unwrap(),
        "2c".parse::<Card>().unwrap(),
    ];
    let rmask = river[0].mask() | river[1].mask() | river[2].mask() | river[3].mask() | river[4].mask();

    let hands: Vec<Hand4> = Hand4::all_hands()
        .map(|(_, h)| h)
        .filter(|h| !h.is_blocked_by(rmask))
        .step_by(20)
        .take(5_000)
        .collect();

    let cache = RiverBoardCache::new(river);
    let rounds = 400usize; // 5,000 * 400 = 2,000,000 full 60-combo river PLO evaluations!
    let t0 = Instant::now();
    let mut checksum = 0u64;
    for _ in 0..rounds {
        for &h in &hands {
            checksum = checksum.wrapping_add(cache.eval_hand_unchecked(h).raw() as u64);
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let total_evals = (hands.len() * rounds) as f64;
    println!(
        "PLO River (60 x 5-card subsets/hand): {:.2}M PLO river hands/sec ({:.1}M 5-card evals/sec), checksum={checksum}",
        (total_evals / elapsed) / 1e6,
        (total_evals * 60.0 / elapsed) / 1e6,
    );

    // Also benchmark standalone eval_plo_river (without prebuilt cache)
    let t1 = Instant::now();
    let mut checksum2 = 0u64;
    for _ in 0..100 {
        for &h in &hands {
            checksum2 = checksum2.wrapping_add(eval_plo_river(h, river).raw() as u64);
        }
    }
    let elapsed2 = t1.elapsed().as_secs_f64();
    let total2 = (hands.len() * 100) as f64;
    println!(
        "PLO River uncached: {:.2}M PLO river hands/sec, checksum={checksum2}",
        (total2 / elapsed2) / 1e6,
    );
}
