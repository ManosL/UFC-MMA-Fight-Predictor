DROP TABLE IF EXISTS "DIM_Gender";

CREATE TABLE IF NOT EXISTS "DIM_Gender" (
    "DIM_Gender_Key"      SERIAL PRIMARY KEY,
    "DIM_Gender_Name"     VARCHAR(100) NOT NULL,
    "__elt_loaded_at"     TIMESTAMP WITH TIME ZONE DEFAULT (TIMEZONE('utc', NOW()))
);