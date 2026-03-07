DROP TABLE IF EXISTS "DIM_Fighter" CASCADE;

CREATE TABLE IF NOT EXISTS "DIM_Fighter" (
    "DIM_Fighter_Key"                               SERIAL UNIQUE,
    "DIM_Fighter_ID"                                VARCHAR(50) NOT NULL,
    "DIM_Fighter_Name"                              VARCHAR(50) NOT NULL,
    "DIM_Fighter_Gender_Key"                        INTEGER NOT NULL,
    "DIM_Fighter_Wins"                              INTEGER NOT NULL,
    "DIM_Fighter_Loses"                             INTEGER NOT NULL,
    "DIM_Fighter_Draws"                             INTEGER NOT NULL,
    "DIM_Fighter_Average_Fight_Time_Minutes"        FLOAT NOT NULL,
    "DIM_Fighter_Height_cm"                         FLOAT,
    "DIM_Fighter_Weight_lbs"                        FLOAT,
    "DIM_Fighter_Reach_cm"                          FLOAT,
    "DIM_Fighter_Stance"                            VARCHAR(50) NOT NULL,
    "DIM_Fighter_Date_of_Birth"                     DATE,
    "DIM_Fighter_Total_UFC_Fights"                  INTEGER NOT NULL,
    "DIM_Fighter_Strikes_Landed_per_Minute"         FLOAT NOT NULL,
    "DIM_Fighter_Striking_Accuracy"                 FLOAT NOT NULL,
    "DIM_Fighter_Strikes_Absorbed_per_Minute"       FLOAT NOT NULL,
    "DIM_Fighter_Striking_Defense"                  FLOAT NOT NULL,
    "DIM_Fighter_Average_Takedowns_per_15_Mins"     FLOAT NOT NULL,
    "DIM_Fighter_Takedown_Defense"                  FLOAT NOT NULL,
    "DIM_Fighter_Takedown_Accuracy"                 FLOAT NOT NULL,
    "DIM_Fighter_Submissions_Average_per_15_Mins"   FLOAT NOT NULL,
    "DIM_Fighter_Effective_From"                    DATE,
    "DIM_Fighter_Effective_Until"                   DATE,
    "__elt_loaded_at"                               TIMESTAMP WITH TIME ZONE DEFAULT (TIMEZONE('utc', NOW())),

    PRIMARY KEY("DIM_Fighter_ID", "DIM_Fighter_Effective_From",
                "DIM_Fighter_Effective_Until"),
    
    FOREIGN KEY ("DIM_Fighter_Gender_Key") REFERENCES "DIM_Gender" ("DIM_Gender_Key")
);