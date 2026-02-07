DROP TABLE IF EXISTS new_fighters_current_stats;

CREATE TABLE new_fighters_current_stats(
    "ID"                            VARCHAR(50),
    "Gender"                        VARCHAR(50) NOT NULL, --
    "Name"                          VARCHAR(50) NOT NULL,
    "Wins"                          INTEGER NOT NULL,
    "Loses"                         INTEGER NOT NULL,
    "Draws"                         INTEGER NOT NULL,
    -- "Average_Fight_Time_Minutes"    FLOAT NOT NULL,
    "Height_cm"                     FLOAT, --
    "Weight_lbs"                    FLOAT,
    "Reach_cm"                      FLOAT, --
    "Stance"                        VARCHAR(50) NOT NULL, --
    "Date_of_Birth"                 DATE,
    "Strikes_Landed_per_Minute"     FLOAT NOT NULL,
    "Striking_Accuracy"             FLOAT NOT NULL,
    "Strikes_Absorbed_per_Minute"   FLOAT NOT NULL,
    "Striking_Defense"              FLOAT NOT NULL,
    "Average_Takedowns"             FLOAT NOT NULL,
    "Takedown_Accuracy"             FLOAT NOT NULL,
    "Takedown_Defense"              FLOAT NOT NULL,
    "Submissions_Average"           FLOAT NOT NULL,

    PRIMARY KEY("ID")
);