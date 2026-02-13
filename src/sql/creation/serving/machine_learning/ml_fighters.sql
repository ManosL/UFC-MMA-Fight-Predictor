DROP VIEW IF EXISTS "ML_Fighters";

CREATE VIEW "ML_Fighters" AS
SELECT
    "DIM_Fighter_ID" AS "Fighter ID",
    "DIM_Fighter_Name" AS "Fighter Name",
    "DIM_Gender_Name" AS "Gender",
    "DIM_Fighter_Wins" AS "Wins",
    "DIM_Fighter_Loses" AS "Loses",
    "DIM_Fighter_Draws" AS "Draws",
    "DIM_Fighter_Average_Fight_Time_Minutes" AS "Avg.Time(in Mins)",
    "DIM_Fighter_Height_cm" AS "Height",
    "DIM_Fighter_Weight_lbs" AS "Weight",
    "DIM_Fighter_Reach_cm" AS "Reach",
    "DIM_Fighter_Stance" AS "Stance",
    "DIM_Fighter_Date_of_Birth" AS "DOB",
    "DIM_Fighter_Strikes_Landed_per_Minute" AS "SLpM",
    "DIM_Fighter_Striking_Accuracy" AS "Str.Acc.",
    "DIM_Fighter_Strikes_Absorbed_per_Minute" AS "SApM",
    "DIM_Fighter_Striking_Defense" AS "Str. Def.",
    "DIM_Fighter_Average_Takedowns_per_15_Mins" AS "TD Avg.",
    "DIM_Fighter_Takedown_Accuracy" AS "TD Acc.",
    "DIM_Fighter_Takedown_Defense" AS "TD Def.",
    "DIM_Fighter_Submissions_Average_per_15_Mins" AS "Sub. Avg."
FROM
    "DIM_Fighter"
INNER JOIN "DIM_Gender"
    ON "DIM_Fighter"."DIM_Fighter_Gender_Key" = "DIM_Gender"."DIM_Gender_Key"
WHERE
    "DIM_Fighter_Effective_Until" = '9999-01-01'::DATE;
