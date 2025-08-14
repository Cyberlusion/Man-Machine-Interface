// Cyberlusion Behavioral Security Engine
// Real-time monitoring and anomaly detection for neural interfaces

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Security event severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

/// Types of security events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    AnomalousPattern { confidence: f64 },
    UnauthorizedAccess { source: String },
    IntegrityViolation { expected: String, actual: String },
    RateLimitExceeded { limit: u32, actual: u32 },
    InjectionAttempt { payload: Vec<u8> },
    HardwareFault { component: String },
    CryptoFailure { operation: String },
    PolicyViolation { policy: String },
}

/// Security event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub id: u64,
    pub timestamp: u64,
    pub event_type: EventType,
    pub severity: Severity,
    pub source: String,
    pub metadata: HashMap<String, String>,
}

/// Behavioral profile for pattern learning
#[derive(Debug, Clone)]
pub struct BehavioralProfile {
    pub user_id: String,
    pub creation_time: Instant,
    pub neural_patterns: NeuralPatternProfile,
    pub command_patterns: CommandPatternProfile,
    pub temporal_patterns: TemporalPatternProfile,
    pub trust_score: f64,
}

/// Neural signal pattern profile
#[derive(Debug, Clone)]
pub struct NeuralPatternProfile {
    pub baseline_power: Vec<f64>,
    pub frequency_distribution: Vec<f64>,
    pub spike_rate_mean: f64,
    pub spike_rate_std: f64,
    pub coherence_matrix: Vec<Vec<f64>>,
    pub entropy_baseline: f64,
}

/// Command pattern profile
#[derive(Debug, Clone)]
pub struct CommandPatternProfile {
    pub command_frequency: HashMap<String, u32>,
    pub command_sequences: Vec<Vec<String>>,
    pub timing_patterns: Vec<Duration>,
    pub error_rate: f64,
}

/// Temporal pattern profile
#[derive(Debug, Clone)]
pub struct TemporalPatternProfile {
    pub active_hours: Vec<bool>, // 24-hour profile
    pub session_durations: VecDeque<Duration>,
    pub inter_command_intervals: VecDeque<Duration>,
    pub circadian_phase: f64,
}

/// Main behavioral security engine
pub struct BehavioralSecurityEngine {
    profiles: Arc<RwLock<HashMap<String, BehavioralProfile>>>,
    events: Arc<Mutex<VecDeque<SecurityEvent>>>,
    rules: Arc<RwLock<Vec<SecurityRule>>>,
    ml_detector: Arc<Mutex<MLAnomalyDetector>>,
    response_system: Arc<Mutex<ResponseSystem>>,
    config: SecurityConfig,
    event_counter: Arc<Mutex<u64>>,
}

/// Security configuration
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    pub max_events: usize,
    pub learning_window: Duration,
    pub anomaly_threshold: f64,
    pub trust_decay_rate: f64,
    pub response_escalation: bool,
    pub forensics_enabled: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        SecurityConfig {
            max_events: 10000,
            learning_window: Duration::from_secs(3600),
            anomaly_threshold: 0.95,
            trust_decay_rate: 0.01,
            response_escalation: true,
            forensics_enabled: true,
        }
    }
}

/// Security rule for policy enforcement
#[derive(Debug, Clone)]
pub struct SecurityRule {
    pub id: String,
    pub name: String,
    pub condition: RuleCondition,
    pub action: RuleAction,
    pub severity: Severity,
    pub enabled: bool,
}

/// Rule conditions
#[derive(Debug, Clone)]
pub enum RuleCondition {
    ThresholdExceeded { metric: String, threshold: f64 },
    PatternMatch { pattern: String },
    TimeWindow { start: u32, end: u32 }, // Hours in 24h format
    FrequencyLimit { max_per_minute: u32 },
    Custom { evaluator: String },
}

/// Rule actions
#[derive(Debug, Clone)]
pub enum RuleAction {
    Log,
    Alert,
    Block,
    RateLimit { limit: u32 },
    RequireAuthentication,
    EnterSafeMode,
    Shutdown,
}

/// Machine learning anomaly detector
pub struct MLAnomalyDetector {
    isolation_forest: IsolationForest,
    autoencoder: Autoencoder,
    lstm_predictor: LSTMPredictor,
    detection_threshold: f64,
}

/// Isolation Forest for anomaly detection
pub struct IsolationForest {
    trees: Vec<IsolationTree>,
    n_samples: usize,
}

/// Isolation Tree node
#[derive(Debug, Clone)]
pub struct IsolationTree {
    split_attr: usize,
    split_value: f64,
    left: Option<Box<IsolationTree>>,
    right: Option<Box<IsolationTree>>,
    size: usize,
    depth: usize,
}

/// Autoencoder for pattern reconstruction
pub struct Autoencoder {
    encoder_weights: Vec<Vec<f64>>,
    decoder_weights: Vec<Vec<f64>>,
    latent_dim: usize,
}

/// LSTM for temporal prediction
pub struct LSTMPredictor {
    weights: LSTMWeights,
    hidden_state: Vec<f64>,
    cell_state: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct LSTMWeights {
    w_forget: Vec<Vec<f64>>,
    w_input: Vec<Vec<f64>>,
    w_candidate: Vec<Vec<f64>>,
    w_output: Vec<Vec<f64>>,
}

/// Response system for security events
pub struct ResponseSystem {
    escalation_level: u32,
    active_responses: Vec<ActiveResponse>,
    response_history: VecDeque<ResponseRecord>,
}

#[derive(Debug, Clone)]
pub struct ActiveResponse {
    pub id: u64,
    pub response_type: RuleAction,
    pub started_at: Instant,
    pub duration: Option<Duration>,
    pub event_id: u64,
}

#[derive(Debug, Clone)]
pub struct ResponseRecord {
    pub response: ActiveResponse,
    pub outcome: ResponseOutcome,
    pub ended_at: Instant,
}

#[derive(Debug, Clone)]
pub enum ResponseOutcome {
    Success,
    Failed { reason: String },
    Escalated,
    UserOverride,
}

impl BehavioralSecurityEngine {
    /// Create new behavioral security engine
    pub fn new(config: SecurityConfig) -> Self {
        BehavioralSecurityEngine {
            profiles: Arc::new(RwLock::new(HashMap::new())),
            events: Arc::new(Mutex::new(VecDeque::with_capacity(config.max_events))),
            rules: Arc::new(RwLock::new(Self::default_rules())),
            ml_detector: Arc::new(Mutex::new(MLAnomalyDetector::new())),
            response_system: Arc::new(Mutex::new(ResponseSystem::new())),
            config,
            event_counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Process neural signal for anomaly detection
    pub fn process_neural_signal(
        &self,
        user_id: &str,
        signal_data: &[f64],
        metadata: HashMap<String, String>,
    ) -> Result<(), SecurityError> {
        // Get or create user profile
        let profile = self.get_or_create_profile(user_id);

        // Extract features from signal
        let features = self.extract_signal_features(signal_data);

        // Check against behavioral profile
        let anomaly_score = self.compute_anomaly_score(&profile, &features)?;

        // ML-based detection
        let ml_score = self.ml_detector.lock().unwrap().detect_anomaly(&features)?;

        // Combined score
        let combined_score = 0.7 * anomaly_score + 0.3 * ml_score;

        // Check if anomalous
        if combined_score > self.config.anomaly_threshold {
            self.handle_anomaly(user_id, combined_score, metadata)?;
        }

        // Update profile
        self.update_profile(user_id, &features)?;

        Ok(())
    }

    /// Process command for security validation
    pub fn validate_command(
        &self,
        user_id: &str,
        command: &str,
        parameters: &HashMap<String, String>,
    ) -> Result<bool, SecurityError> {
        // Check rate limiting
        if !self.check_rate_limit(user_id, command)? {
            self.log_event(
                EventType::RateLimitExceeded {
                    limit: 10,
                    actual: 11,
                },
                Severity::Medium,
                user_id.to_string(),
            )?;
            return Ok(false);
        }

        // Check command injection
        if self.detect_injection(command, parameters)? {
            self.log_event(
                EventType::InjectionAttempt {
                    payload: command.as_bytes().to_vec(),
                },
                Severity::High,
                user_id.to_string(),
            )?;
            return Ok(false);
        }

        // Check against security rules
        let rules = self.rules.read().unwrap();
        for rule in rules.iter() {
            if rule.enabled && self.evaluate_rule(rule, user_id, command)? {
                self.execute_rule_action(&rule.action, user_id)?;

                if matches!(rule.action, RuleAction::Block) {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Extract features from neural signal
    fn extract_signal_features(&self, signal: &[f64]) -> SignalFeatures {
        let mean = signal.iter().sum::<f64>() / signal.len() as f64;
        let variance = signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / signal.len() as f64;

        let entropy = self.calculate_entropy(signal);
        let power_spectrum = self.compute_power_spectrum(signal);
        let spike_rate = self.estimate_spike_rate(signal);

        SignalFeatures {
            mean,
            variance,
            entropy,
            power_spectrum,
            spike_rate,
            raw_samples: signal.to_vec(),
        }
    }

    /// Calculate Shannon entropy
    fn calculate_entropy(&self, signal: &[f64]) -> f64 {
        let mut histogram = HashMap::new();
        let bin_size = 0.1;

        for &value in signal {
            let bin = (value / bin_size).floor() as i32;
            *histogram.entry(bin).or_insert(0) += 1;
        }

        let total = signal.len() as f64;
        let mut entropy = 0.0;

        for count in histogram.values() {
            let p = *count as f64 / total;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Compute power spectrum using FFT
    fn compute_power_spectrum(&self, signal: &[f64]) -> Vec<f64> {
        // Simplified FFT implementation
        // In production, use proper FFT library
        let n = signal.len();
        let mut spectrum = vec![0.0; n / 2];

        for k in 0..n / 2 {
            let mut real = 0.0;
            let mut imag = 0.0;

            for (i, &x) in signal.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
                real += x * angle.cos();
                imag += x * angle.sin();
            }

            spectrum[k] = (real * real + imag * imag).sqrt();
        }

        spectrum
    }

    /// Estimate spike rate from signal
    fn estimate_spike_rate(&self, signal: &[f64]) -> f64 {
        let threshold =
            signal.iter().sum::<f64>() / signal.len() as f64 - 3.0 * self.calculate_std(signal);

        let mut spike_count = 0;
        let mut in_spike = false;

        for &value in signal {
            if value < threshold && !in_spike {
                spike_count += 1;
                in_spike = true;
            } else if value > threshold {
                in_spike = false;
            }
        }

        spike_count as f64 / (signal.len() as f64 / 30000.0) // Assuming 30kHz sampling
    }

    /// Calculate standard deviation
    fn calculate_std(&self, signal: &[f64]) -> f64 {
        let mean = signal.iter().sum::<f64>() / signal.len() as f64;
        let variance = signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / signal.len() as f64;
        variance.sqrt()
    }

    /// Compute anomaly score
    fn compute_anomaly_score(
        &self,
        profile: &BehavioralProfile,
        features: &SignalFeatures,
    ) -> Result<f64, SecurityError> {
        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // Compare entropy
        let entropy_diff = (features.entropy - profile.neural_patterns.entropy_baseline).abs();
        let entropy_score = 1.0 - (-entropy_diff).exp();
        score += entropy_score * 0.3;
        weight_sum += 0.3;

        // Compare spike rate
        let spike_z_score = (features.spike_rate - profile.neural_patterns.spike_rate_mean)
            / profile.neural_patterns.spike_rate_std.max(0.001);
        let spike_score = 1.0 - (-spike_z_score.abs() / 2.0).exp();
        score += spike_score * 0.3;
        weight_sum += 0.3;

        // Compare power spectrum
        if !profile.neural_patterns.baseline_power.is_empty() {
            let power_distance = self.calculate_distance(
                &features.power_spectrum,
                &profile.neural_patterns.baseline_power,
            );
            let power_score = 1.0 - (-power_distance / 10.0).exp();
            score += power_score * 0.4;
            weight_sum += 0.4;
        }

        Ok(score / weight_sum)
    }

    /// Calculate Euclidean distance
    fn calculate_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let min_len = a.len().min(b.len());
        a.iter()
            .zip(b.iter())
            .take(min_len)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Check rate limiting
    fn check_rate_limit(&self, user_id: &str, command: &str) -> Result<bool, SecurityError> {
        // Simplified rate limiting
        // In production, use proper rate limiting with sliding windows
        Ok(true)
    }

    /// Detect command injection attempts
    fn detect_injection(
        &self,
        command: &str,
        parameters: &HashMap<String, String>,
    ) -> Result<bool, SecurityError> {
        // Check for suspicious patterns
        let suspicious_patterns = [
            "';", "--;", "/*", "*/", "xp_", "sp_", "exec", "execute", "drop", "alter", "union",
            "../", "..\\", "%00", "\0", "\\x00",
        ];

        for pattern in &suspicious_patterns {
            if command.contains(pattern) {
                return Ok(true);
            }

            for value in parameters.values() {
                if value.contains(pattern) {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Get or create user profile
    fn get_or_create_profile(&self, user_id: &str) -> BehavioralProfile {
        let mut profiles = self.profiles.write().unwrap();

        profiles
            .entry(user_id.to_string())
            .or_insert_with(|| BehavioralProfile {
                user_id: user_id.to_string(),
                creation_time: Instant::now(),
                neural_patterns: NeuralPatternProfile {
                    baseline_power: vec![],
                    frequency_distribution: vec![],
                    spike_rate_mean: 10.0,
                    spike_rate_std: 2.0,
                    coherence_matrix: vec![],
                    entropy_baseline: 3.0,
                },
                command_patterns: CommandPatternProfile {
                    command_frequency: HashMap::new(),
                    command_sequences: vec![],
                    timing_patterns: vec![],
                    error_rate: 0.0,
                },
                temporal_patterns: TemporalPatternProfile {
                    active_hours: vec![false; 24],
                    session_durations: VecDeque::new(),
                    inter_command_intervals: VecDeque::new(),
                    circadian_phase: 0.0,
                },
                trust_score: 0.5,
            })
            .clone()
    }

    /// Update user profile with new features
    fn update_profile(
        &self,
        user_id: &str,
        features: &SignalFeatures,
    ) -> Result<(), SecurityError> {
        let mut profiles = self.profiles.write().unwrap();

        if let Some(profile) = profiles.get_mut(user_id) {
            // Update neural patterns (exponential moving average)
            let alpha = 0.1;

            profile.neural_patterns.entropy_baseline =
                (1.0 - alpha) * profile.neural_patterns.entropy_baseline + alpha * features.entropy;

            profile.neural_patterns.spike_rate_mean = (1.0 - alpha)
                * profile.neural_patterns.spike_rate_mean
                + alpha * features.spike_rate;

            // Update trust score
            profile.trust_score = (profile.trust_score * (1.0 - self.config.trust_decay_rate))
                .max(0.0)
                .min(1.0);
        }

        Ok(())
    }

    /// Handle detected anomaly
    fn handle_anomaly(
        &self,
        user_id: &str,
        score: f64,
        metadata: HashMap<String, String>,
    ) -> Result<(), SecurityError> {
        let severity = if score > 0.99 {
            Severity::Critical
        } else if score > 0.95 {
            Severity::High
        } else if score > 0.90 {
            Severity::Medium
        } else {
            Severity::Low
        };

        self.log_event(
            EventType::AnomalousPattern { confidence: score },
            severity,
            user_id.to_string(),
        )?;

        // Trigger response based on severity
        if severity >= Severity::High {
            let mut response_system = self.response_system.lock().unwrap();
            response_system.trigger_response(severity)?;
        }

        Ok(())
    }

    /// Log security event
    fn log_event(
        &self,
        event_type: EventType,
        severity: Severity,
        source: String,
    ) -> Result<(), SecurityError> {
        let mut counter = self.event_counter.lock().unwrap();
        *counter += 1;
        let event_id = *counter;

        let event = SecurityEvent {
            id: event_id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            event_type,
            severity,
            source,
            metadata: HashMap::new(),
        };

        let mut events = self.events.lock().unwrap();
        events.push_back(event.clone());

        // Maintain max events
        while events.len() > self.config.max_events {
            events.pop_front();
        }

        // Log to persistent storage in production
        println!("Security Event: {:?}", event);

        Ok(())
    }

    /// Evaluate security rule
    fn evaluate_rule(
        &self,
        rule: &SecurityRule,
        user_id: &str,
        command: &str,
    ) -> Result<bool, SecurityError> {
        match &rule.condition {
            RuleCondition::ThresholdExceeded { metric, threshold } => {
                // Check if metric exceeds threshold
                // Implementation depends on specific metrics
                Ok(false)
            }
            RuleCondition::PatternMatch { pattern } => Ok(command.contains(pattern)),
            RuleCondition::TimeWindow { start, end } => {
                let now = chrono::Local::now();
                let hour = now.hour();
                Ok(hour >= *start && hour <= *end)
            }
            RuleCondition::FrequencyLimit { max_per_minute } => {
                // Check command frequency
                // Implementation with sliding window
                Ok(false)
            }
            RuleCondition::Custom { evaluator } => {
                // Custom evaluation logic
                Ok(false)
            }
        }
    }

    /// Execute rule action
    fn execute_rule_action(&self, action: &RuleAction, user_id: &str) -> Result<(), SecurityError> {
        match action {
            RuleAction::Log => {
                println!("Rule action: Log for user {}", user_id);
            }
            RuleAction::Alert => {
                println!("Rule action: Alert for user {}", user_id);
            }
            RuleAction::Block => {
                println!("Rule action: Block for user {}", user_id);
            }
            RuleAction::RateLimit { limit } => {
                println!("Rule action: Rate limit {} for user {}", limit, user_id);
            }
            RuleAction::RequireAuthentication => {
                println!("Rule action: Require auth for user {}", user_id);
            }
            RuleAction::EnterSafeMode => {
                println!("Rule action: Enter safe mode for user {}", user_id);
            }
            RuleAction::Shutdown => {
                println!("Rule action: Shutdown for user {}", user_id);
            }
        }

        Ok(())
    }

    /// Default security rules
    fn default_rules() -> Vec<SecurityRule> {
        vec![
            SecurityRule {
                id: "rule_001".to_string(),
                name: "High frequency commands".to_string(),
                condition: RuleCondition::FrequencyLimit { max_per_minute: 60 },
                action: RuleAction::RateLimit { limit: 30 },
                severity: Severity::Medium,
                enabled: true,
            },
            SecurityRule {
                id: "rule_002".to_string(),
                name: "Nighttime restriction".to_string(),
                condition: RuleCondition::TimeWindow { start: 2, end: 5 },
                action: RuleAction::RequireAuthentication,
                severity: Severity::Low,
                enabled: true,
            },
        ]
    }
}

/// Signal features structure
#[derive(Debug, Clone)]
struct SignalFeatures {
    mean: f64,
    variance: f64,
    entropy: f64,
    power_spectrum: Vec<f64>,
    spike_rate: f64,
    raw_samples: Vec<f64>,
}

/// Custom error type
#[derive(Debug)]
pub enum SecurityError {
    ProfileNotFound,
    DetectionFailed,
    ResponseFailed,
    ConfigError,
}

impl std::fmt::Display for SecurityError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SecurityError::ProfileNotFound => write!(f, "Profile not found"),
            SecurityError::DetectionFailed => write!(f, "Detection failed"),
            SecurityError::ResponseFailed => write!(f, "Response failed"),
            SecurityError::ConfigError => write!(f, "Configuration error"),
        }
    }
}

impl std::error::Error for SecurityError {}

// ML Anomaly Detector implementations
impl MLAnomalyDetector {
    fn new() -> Self {
        MLAnomalyDetector {
            isolation_forest: IsolationForest::new(100),
            autoencoder: Autoencoder::new(100, 10),
            lstm_predictor: LSTMPredictor::new(100, 50),
            detection_threshold: 0.95,
        }
    }

    fn detect_anomaly(&self, features: &SignalFeatures) -> Result<f64, SecurityError> {
        // Simplified anomaly detection
        // In production, use proper ML models
        Ok(0.5)
    }
}

impl IsolationForest {
    fn new(n_trees: usize) -> Self {
        IsolationForest {
            trees: vec![],
            n_samples: 0,
        }
    }
}

impl Autoencoder {
    fn new(input_dim: usize, latent_dim: usize) -> Self {
        Autoencoder {
            encoder_weights: vec![vec![0.0; latent_dim]; input_dim],
            decoder_weights: vec![vec![0.0; input_dim]; latent_dim],
            latent_dim,
        }
    }
}

impl LSTMPredictor {
    fn new(input_dim: usize, hidden_dim: usize) -> Self {
        LSTMPredictor {
            weights: LSTMWeights {
                w_forget: vec![vec![0.0; hidden_dim]; input_dim + hidden_dim],
                w_input: vec![vec![0.0; hidden_dim]; input_dim + hidden_dim],
                w_candidate: vec![vec![0.0; hidden_dim]; input_dim + hidden_dim],
                w_output: vec![vec![0.0; hidden_dim]; input_dim + hidden_dim],
            },
            hidden_state: vec![0.0; hidden_dim],
            cell_state: vec![0.0; hidden_dim],
        }
    }
}

impl ResponseSystem {
    fn new() -> Self {
        ResponseSystem {
            escalation_level: 0,
            active_responses: vec![],
            response_history: VecDeque::new(),
        }
    }

    fn trigger_response(&mut self, severity: Severity) -> Result<(), SecurityError> {
        // Implement response logic based on severity
        self.escalation_level = match severity {
            Severity::Critical => 5,
            Severity::High => 4,
            Severity::Medium => 3,
            Severity::Low => 2,
            Severity::Info => 1,
        };

        Ok(())
    }
}

// Use statements for external dependencies
use chrono::Timelike;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_engine_creation() {
        let engine = BehavioralSecurityEngine::new(SecurityConfig::default());
        assert!(engine.profiles.read().unwrap().is_empty());
    }

    #[test]
    fn test_anomaly_detection() {
        let engine = BehavioralSecurityEngine::new(SecurityConfig::default());
        let signal = vec![0.0; 1000];
        let result = engine.process_neural_signal("test_user", &signal, HashMap::new());
        assert!(result.is_ok());
    }

    #[test]
    fn test_command_validation() {
        let engine = BehavioralSecurityEngine::new(SecurityConfig::default());
        let result = engine.validate_command("test_user", "move_forward", &HashMap::new());
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_injection_detection() {
        let engine = BehavioralSecurityEngine::new(SecurityConfig::default());
        let mut params = HashMap::new();
        params.insert("input".to_string(), "'; DROP TABLE users;--".to_string());
        let result = engine.validate_command("test_user", "process", &params);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }
}
