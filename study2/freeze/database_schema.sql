PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS slots(
  scientific_id TEXT PRIMARY KEY, plan_index INTEGER NOT NULL UNIQUE, item_id TEXT NOT NULL,
  group_id TEXT NOT NULL, provider TEXT NOT NULL, model_key TEXT NOT NULL, role TEXT NOT NULL,
  condition TEXT, parent_initial_id TEXT, gold_label TEXT NOT NULL, target_label TEXT,
  status TEXT NOT NULL, request_json TEXT, payload_json TEXT, payload_sha256 TEXT,
  committed_attempt_id INTEGER, ineligible_reason TEXT, updated_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts(
  attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, scientific_id TEXT NOT NULL,
  slot_attempt INTEGER NOT NULL, replacement_of_attempt_id INTEGER, provider TEXT NOT NULL,
  status TEXT NOT NULL, classification TEXT, payload_sha256 TEXT NOT NULL,
  payload_json TEXT NOT NULL, reservation_usd TEXT NOT NULL, started_utc TEXT NOT NULL,
  started_epoch REAL NOT NULL, response_utc TEXT, safe_http_json TEXT, raw_response_json TEXT,
  failure_json TEXT, UNIQUE(scientific_id,slot_attempt),
  FOREIGN KEY(scientific_id) REFERENCES slots(scientific_id)
);
CREATE TABLE IF NOT EXISTS observations(
  scientific_id TEXT PRIMARY KEY, attempt_id INTEGER NOT NULL UNIQUE, item_id TEXT NOT NULL,
  group_id TEXT NOT NULL, provider TEXT NOT NULL, model_key TEXT NOT NULL, role TEXT NOT NULL,
  condition TEXT, parent_initial_id TEXT, target_label TEXT, gold_label TEXT NOT NULL,
  request_json TEXT NOT NULL, payload_json TEXT NOT NULL, raw_response_json TEXT NOT NULL,
  visible_text TEXT, parsed_status TEXT NOT NULL, parsed_label TEXT, invalid_reason TEXT,
  returned_model TEXT, provider_request_id TEXT, input_tokens INTEGER, output_tokens INTEGER,
  reasoning_tokens INTEGER, cached_tokens INTEGER, total_tokens INTEGER, latency_seconds REAL,
  estimated_cost_usd TEXT NOT NULL, committed_utc TEXT NOT NULL,
  FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id)
);
CREATE TABLE IF NOT EXISTS events(
  event_id INTEGER PRIMARY KEY AUTOINCREMENT, utc TEXT NOT NULL, scientific_id TEXT,
  provider TEXT, kind TEXT NOT NULL, details_json TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS observations_no_update BEFORE UPDATE ON observations
BEGIN SELECT RAISE(ABORT,'immutable observation'); END;
CREATE TRIGGER IF NOT EXISTS observations_no_delete BEFORE DELETE ON observations
BEGIN SELECT RAISE(ABORT,'immutable observation'); END;
