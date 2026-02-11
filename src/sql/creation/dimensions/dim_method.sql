DROP TABLE IF EXISTS "DIM_Method";

CREATE TABLE IF NOT EXISTS "DIM_Method" (
    "DIM_Method_Key"      SERIAL PRIMARY KEY,
    "DIM_Method_Name"     VARCHAR(100) NOT NULL,
    "__elt_loaded_at"     TIMESTAMP WITH TIME ZONE DEFAULT (TIMEZONE('utc', NOW()))
);