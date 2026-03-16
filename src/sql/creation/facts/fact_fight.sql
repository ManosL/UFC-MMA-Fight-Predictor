DROP TABLE IF EXISTS "FACT_Fight" CASCADE;

CREATE TABLE IF NOT EXISTS "FACT_Fight" (
    "FACT_Fight_Key"                                   SERIAL PRIMARY KEY,
    "FACT_Fight_ID"                                    VARCHAR(50),
    "FACT_Fight_Date_Key"                              VARCHAR(10),
    "FACT_Fight_Gender_Key"                            INTEGER,
    "FACT_Fight_Weight_Class_Key"                      INTEGER,
    "FACT_Fight_Title_Fight"                           BOOLEAN,
    "FACT_Fight_Method_Key"                            INTEGER,
    "FACT_Fight_Round"                                 INTEGER,
    "FACT_Fight_Time"                                  VARCHAR(10),
    "FACT_Fight_Fight_Time_Format_Key"                 INTEGER,
    "FACT_Fight_Duration_Mins"                         FLOAT,
    "__elt_loaded_at"                                  TIMESTAMP WITH TIME ZONE DEFAULT (TIMEZONE('utc', NOW())),

    FOREIGN KEY ("FACT_Fight_Date_Key") REFERENCES "DIM_Date" ("DIM_Date_Key"),
    FOREIGN KEY ("FACT_Fight_Gender_Key") REFERENCES "DIM_Gender" ("DIM_Gender_Key"),
    FOREIGN KEY ("FACT_Fight_Weight_Class_Key") REFERENCES "DIM_Weight_Class" ("DIM_Weight_Class_Key"),
    FOREIGN KEY ("FACT_Fight_Method_Key") REFERENCES "DIM_Method" ("DIM_Method_Key"),
    FOREIGN KEY ("FACT_Fight_Fight_Time_Format_Key") REFERENCES "DIM_Fight_Time_Format" ("DIM_Fight_Time_Format_Key")
);