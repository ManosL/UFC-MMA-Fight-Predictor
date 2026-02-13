DROP TABLE IF EXISTS "DIM_Fight_Time_Format" CASCADE;

CREATE TABLE IF NOT EXISTS "DIM_Fight_Time_Format" (
    "DIM_Fight_Time_Format_Key"      SERIAL PRIMARY KEY,
    "DIM_Fight_Time_Format_Name"     VARCHAR(100) NOT NULL,
    "__elt_loaded_at"                TIMESTAMP WITH TIME ZONE DEFAULT (TIMEZONE('utc', NOW()))
);