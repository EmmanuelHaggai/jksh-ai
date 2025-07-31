use crate::error::{AppError, Result};
use crate::models::*;
use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::cmp::Ordering;
use rayon::prelude::*;
use chrono::{DateTime, Utc};
use sha2::{Sha256, Digest};
use std::time::Instant;

// Complex mathematical computations
pub struct MathUtils;

impl MathUtils {
    pub fn fibonacci_matrix(n: u64) -> Result<u64> {
        if n == 0 { return Ok(0); }
        if n == 1 { return Ok(1); }

        let mut result = [[1u64, 0], [0, 1]]; // Identity matrix
        let mut base = [[1u64, 1], [1, 0]];   // Fibonacci matrix
        let mut exp = n - 1;

        while exp > 0 {
            if exp % 2 == 1 {
                result = Self::matrix_multiply(result, base);
            }
            base = Self::matrix_multiply(base, base);
            exp /= 2;
        }

        Ok(result[0][0])
    }

    fn matrix_multiply(a: [[u64; 2]; 2], b: [[u64; 2]; 2]) -> [[u64; 2]; 2] {
        [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
        ]
    }

    pub fn prime_factorization(mut n: u64) -> Vec<u64> {
        let mut factors = Vec::new();
        
        // Handle 2 separately
        while n % 2 == 0 {
            factors.push(2);
            n /= 2;
        }

        // Check odd numbers from 3 onwards
        let mut i = 3;
        while i * i <= n {
            while n % i == 0 {
                factors.push(i);
                n /= i;
            }
            i += 2;
        }

        if n > 2 {
            factors.push(n);
        }

        factors
    }

    pub fn monte_carlo_pi(iterations: u64) -> f64 {
        let inside_circle: u64 = (0..iterations)
            .into_par_iter()
            .map(|_| {
                let x: f64 = rand::random::<f64>() * 2.0 - 1.0;
                let y: f64 = rand::random::<f64>() * 2.0 - 1.0;
                if x * x + y * y <= 1.0 { 1 } else { 0 }
            })
            .sum();

        4.0 * inside_circle as f64 / iterations as f64
    }

    pub fn complex_polynomial_evaluation(coefficients: &[f64], x: f64) -> f64 {
        coefficients.iter().enumerate().map(|(i, &coef)| coef * x.powi(i as i32)).sum()
    }

    pub fn numerical_integration_simpson(
        f: fn(f64) -> f64,
        a: f64,
        b: f64,
        n: usize,
    ) -> Result<f64> {
        if n % 2 != 0 {
            return Err(AppError::Computation("n must be even for Simpson's rule".to_string()));
        }

        let h = (b - a) / n as f64;
        let mut sum = f(a) + f(b);

        for i in 1..n {
            let x = a + i as f64 * h;
            if i % 2 == 0 {
                sum += 2.0 * f(x);
            } else {
                sum += 4.0 * f(x);
            }
        }

        Ok(sum * h / 3.0)
    }
}

// Machine Learning utilities
pub struct MLUtils;

impl MLUtils {
    pub fn linear_regression(training_data: &[TrainingExample]) -> Result<(f64, f64)> {
        if training_data.is_empty() {
            return Err(AppError::Computation("No training data provided".to_string()));
        }

        let n = training_data.len() as f64;
        let sum_x: f64 = training_data.iter().map(|ex| ex.features[0]).sum();
        let sum_y: f64 = training_data.iter().map(|ex| ex.target).sum();
        let sum_xy: f64 = training_data.iter().map(|ex| ex.features[0] * ex.target).sum();
        let sum_x_squared: f64 = training_data.iter().map(|ex| ex.features[0] * ex.features[0]).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        Ok((slope, intercept))
    }

    pub fn k_means_clustering(data: &[Vec<f64>], k: usize, max_iterations: usize) -> Result<Vec<Vec<f64>>> {
        if data.is_empty() || k == 0 {
            return Err(AppError::Computation("Invalid input for k-means".to_string()));
        }

        let dimensions = data[0].len();
        let mut centroids: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..dimensions).map(|_| rand::random::<f64>()).collect())
            .collect();

        for _ in 0..max_iterations {
            let mut clusters: Vec<Vec<Vec<f64>>> = vec![Vec::new(); k];

            // Assign points to nearest centroid
            for point in data {
                let mut min_distance = f64::INFINITY;
                let mut closest_centroid = 0;

                for (i, centroid) in centroids.iter().enumerate() {
                    let distance = Self::euclidean_distance(point, centroid);
                    if distance < min_distance {
                        min_distance = distance;
                        closest_centroid = i;
                    }
                }

                clusters[closest_centroid].push(point.clone());
            }

            // Update centroids
            let mut new_centroids = Vec::new();
            for cluster in &clusters {
                if cluster.is_empty() {
                    new_centroids.push(centroids[new_centroids.len()].clone());
                } else {
                    let mut centroid = vec![0.0; dimensions];
                    for point in cluster {
                        for (i, &val) in point.iter().enumerate() {
                            centroid[i] += val;
                        }
                    }
                    for val in &mut centroid {
                        *val /= cluster.len() as f64;
                    }
                    new_centroids.push(centroid);
                }
            }

            centroids = new_centroids;
        }

        Ok(centroids)
    }

    fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    pub fn neural_network_forward_pass(
        inputs: &[f64],
        weights: &[Vec<Vec<f64>>],
        biases: &[Vec<f64>],
    ) -> Result<Vec<f64>> {
        let mut activations = inputs.to_vec();

        for (layer_weights, layer_biases) in weights.iter().zip(biases.iter()) {
            let mut new_activations = Vec::new();

            for (neuron_weights, &bias) in layer_weights.iter().zip(layer_biases.iter()) {
                let weighted_sum: f64 = activations
                    .iter()
                    .zip(neuron_weights.iter())
                    .map(|(&a, &w)| a * w)
                    .sum::<f64>() + bias;

                new_activations.push(Self::sigmoid(weighted_sum));
            }

            activations = new_activations;
        }

        Ok(activations)
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

// Graph algorithms
pub struct GraphUtils;

#[derive(Debug, Clone)]
struct Edge {
    to: String,
    weight: f64,
}

#[derive(Debug, Clone)]
struct Node {
    id: String,
    distance: f64,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Node {}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance) // Reverse for min-heap
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl GraphUtils {
    pub fn dijkstra_shortest_path(
        nodes: &[GraphNode],
        edges: &[GraphEdge],
        start: &str,
        end: &str,
    ) -> Result<(Vec<String>, f64)> {
        let mut graph: HashMap<String, Vec<Edge>> = HashMap::new();
        
        // Build adjacency list
        for node in nodes {
            graph.insert(node.id.clone(), Vec::new());
        }
        
        for edge in edges {
            graph.entry(edge.from.clone())
                .or_default()
                .push(Edge {
                    to: edge.to.clone(),
                    weight: edge.weight,
                });
        }

        let mut distances: HashMap<String, f64> = HashMap::new();
        let mut previous: HashMap<String, Option<String>> = HashMap::new();
        let mut heap = BinaryHeap::new();

        // Initialize distances
        for node in nodes {
            distances.insert(node.id.clone(), f64::INFINITY);
            previous.insert(node.id.clone(), None);
        }
        distances.insert(start.to_string(), 0.0);
        heap.push(Node {
            id: start.to_string(),
            distance: 0.0,
        });

        while let Some(current) = heap.pop() {
            if current.id == end {
                break;
            }

            if current.distance > distances[&current.id] {
                continue;
            }

            if let Some(neighbors) = graph.get(&current.id) {
                for edge in neighbors {
                    let distance = current.distance + edge.weight;

                    if distance < distances[&edge.to] {
                        distances.insert(edge.to.clone(), distance);
                        previous.insert(edge.to.clone(), Some(current.id.clone()));
                        heap.push(Node {
                            id: edge.to.clone(),
                            distance,
                        });
                    }
                }
            }
        }

        // Reconstruct path
        let mut path = Vec::new();
        let mut current = Some(end.to_string());
        
        while let Some(node) = current {
            path.push(node.clone());
            current = previous[&node].clone();
        }
        
        if path.last() != Some(&start.to_string()) {
            return Err(AppError::Computation("No path found".to_string()));
        }
        
        path.reverse();
        let total_distance = distances[end];

        Ok((path, total_distance))
    }

    pub fn traveling_salesman_genetic_algorithm(
        cities: &[GraphNode],
        distance_matrix: &[Vec<f64>],
        population_size: usize,
        generations: usize,
    ) -> Result<(Vec<String>, f64, u64)> {
        if cities.is_empty() {
            return Err(AppError::Computation("No cities provided".to_string()));
        }

        let city_count = cities.len();
        let mut population: Vec<Vec<usize>> = Vec::new();

        // Initialize random population
        for _ in 0..population_size {
            let mut route: Vec<usize> = (0..city_count).collect();
            route.shuffle(&mut rand::thread_rng());
            population.push(route);
        }

        let mut best_route = population[0].clone();
        let mut best_distance = Self::calculate_route_distance(&best_route, distance_matrix);

        for generation in 0..generations {
            // Selection, crossover, and mutation would go here
            // For simplicity, we'll just do random improvements
            
            for route in &mut population {
                // Random 2-opt improvement
                if city_count > 3 {
                    let i = rand::random::<usize>() % city_count;
                    let j = rand::random::<usize>() % city_count;
                    if i != j {
                        route.swap(i, j);
                        let distance = Self::calculate_route_distance(route, distance_matrix);
                        if distance < best_distance {
                            best_distance = distance;
                            best_route = route.clone();
                        }
                    }
                }
            }
        }

        let route_names: Vec<String> = best_route
            .iter()
            .map(|&i| cities[i].id.clone())
            .collect();

        Ok((route_names, best_distance, generations as u64))
    }

    fn calculate_route_distance(route: &[usize], distance_matrix: &[Vec<f64>]) -> f64 {
        let mut total = 0.0;
        for i in 0..route.len() {
            let from = route[i];
            let to = route[(i + 1) % route.len()];
            total += distance_matrix[from][to];
        }
        total
    }
}

// Blockchain utilities
pub struct BlockchainUtils;

impl BlockchainUtils {
    pub fn validate_blockchain(blocks: &[BlockchainBlock]) -> Result<(bool, Vec<u64>)> {
        let mut invalid_blocks = Vec::new();

        for (i, block) in blocks.iter().enumerate() {
            if i == 0 {
                // Genesis block validation
                if block.previous_hash != "0" {
                    invalid_blocks.push(block.index);
                }
            } else {
                // Validate previous hash
                if block.previous_hash != blocks[i - 1].hash {
                    invalid_blocks.push(block.index);
                }
            }

            // Validate block hash
            let calculated_hash = Self::calculate_block_hash(block);
            if calculated_hash != block.hash {
                invalid_blocks.push(block.index);
            }
        }

        Ok((invalid_blocks.is_empty(), invalid_blocks))
    }

    pub fn calculate_block_hash(block: &BlockchainBlock) -> String {
        let data = format!(
            "{}{}{}{}{}",
            block.index, block.timestamp, block.data, block.previous_hash, block.nonce
        );
        
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn mine_block(
        index: u64,
        data: &str,
        previous_hash: &str,
        difficulty: usize,
    ) -> Result<BlockchainBlock> {
        let timestamp = Utc::now();
        let target = "0".repeat(difficulty);
        let mut nonce = 0u64;

        loop {
            let block = BlockchainBlock {
                index,
                timestamp,
                data: data.to_string(),
                previous_hash: previous_hash.to_string(),
                hash: String::new(),
                nonce,
            };

            let hash = Self::calculate_block_hash(&block);
            
            if hash.starts_with(&target) {
                return Ok(BlockchainBlock {
                    index,
                    timestamp,
                    data: data.to_string(),
                    previous_hash: previous_hash.to_string(),
                    hash,
                    nonce,
                });
            }

            nonce += 1;
            
            if nonce > 1_000_000 {
                return Err(AppError::Computation("Mining timeout".to_string()));
            }
        }
    }
}

// Cryptographic utilities
pub struct CryptoUtils;

impl CryptoUtils {
    pub fn sha256_hash(data: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn merkle_tree_root(data: &[String]) -> Result<String> {
        if data.is_empty() {
            return Err(AppError::Computation("No data for merkle tree".to_string()));
        }

        let mut hashes: Vec<String> = data.iter().map(|d| Self::sha256_hash(d)).collect();

        while hashes.len() > 1 {
            let mut next_level = Vec::new();
            
            for chunk in hashes.chunks(2) {
                let combined = if chunk.len() == 2 {
                    format!("{}{}", chunk[0], chunk[1])
                } else {
                    format!("{}{}", chunk[0], chunk[0])
                };
                next_level.push(Self::sha256_hash(&combined));
            }
            
            hashes = next_level;
        }

        Ok(hashes[0].clone())
    }
}

// Performance monitoring utilities
pub struct PerformanceMonitor {
    start_time: Instant,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }

    pub fn elapsed_ms(&self) -> u128 {
        self.start_time.elapsed().as_millis()
    }
}

// Additional complex utilities
use rand::seq::SliceRandom;

trait Shuffle<T> {
    fn shuffle(&mut self, rng: &mut impl rand::Rng);
}

impl<T> Shuffle<T> for Vec<T> {
    fn shuffle(&mut self, rng: &mut impl rand::Rng) {
        self.as_mut_slice().shuffle(rng);
    }
}