-- Normalize agent JSON fields to valid defaults
-- Required for Streamlit caching (pickle-safe)

UPDATE agents
SET input_vars = '[]'
WHERE input_vars IS NULL OR TRIM(input_vars) = '';

UPDATE agents
SET output_vars = '[]'
WHERE output_vars IS NULL OR TRIM(output_vars) = '';