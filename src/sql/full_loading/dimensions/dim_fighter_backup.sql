INSERT INTO "DIM_Fighter" (
    "DIM_Fighter_ID",
    "DIM_Fighter_Name",
    "DIM_Fighter_Age",
    "DIM_Fighter_Wins",
    "DIM_Fighter_Loses",
    "DIM_Fighter_Draws",
    "DIM_Fighter_Average_Fight_Time_Minutes",
    "DIM_Fighter_Height_cm",
    "DIM_Fighter_Weight_lbs",
    "DIM_Fighter_Reach_cm",
    "DIM_Fighter_Stance",
    "DIM_Fighter_Date_of_Birth",
    "DIM_Fighter_Strikes_Landed_per_Minute",
    "DIM_Fighter_Striking_Accuracy",
    "DIM_Fighter_Strikes_Absorbed_per_Minute",
    "DIM_Fighter_Striking_Defense",
    "DIM_Fighter_Average_Takedowns_per_15_Mins",
    "DIM_Fighter_Takedown_Defense",
    "DIM_Fighter_Takedown_Accuracy",
    "DIM_Fighter_Submissions_Average_per_15_Mins",
    "DIM_Fighter_Effective_From",
    "DIM_Fighter_Effective_Until"
)
WITH fighter_stats_before_fights AS (
    SELECT
        "Fight_ID",
        "Fighter_1_ID" AS "ID",
        "Fighter_1_Name" AS "Name",
        "Fighter_1_Age" AS "Age",
        "Fighter_1_Wins" AS "Wins",
        "Fighter_1_Loses" AS "Loses",
        "Fighter_1_Draws" AS "Draws",
        "Fighter_1_Avg_Fight_Time_Mins" AS "Avg_Fight_Time_Mins",
        "Fighter_1_Sign_Strikes_Landed_per_Min" AS "Sign_Strikes_Landed_per_Min",
        "Fighter_1_Striking_Accuracy" AS "Striking_Accuracy",
        "Fighter_1_Sign_Strikes_Absorbed_per_Min" AS "Sign_Strikes_Absorbed_per_Min",
        "Fighter_1_Striking_Defense" AS "Striking_Defense",
        "Fighter_1_Takedown_Average_per_15Min" AS "Takedown_Average_per_15Min",
        "Fighter_1_Takedown_Accuracy" AS "Takedown_Accuracy",
        "Fighter_1_Takedown_Defense" AS "Takedown_Defense",
        "Fighter_1_Submission_Average_per_15M" AS "Submission_Average_per_15M"
    FROM
        raw_fighters_stats_before_fight
    UNION
    SELECT
        "Fight_ID",
        "Fighter_2_ID" AS "ID",
        "Fighter_2_Name" AS "Name",
        "Fighter_2_Age" AS "Age",
        "Fighter_2_Wins" AS "Wins",
        "Fighter_2_Loses" AS "Loses",
        "Fighter_2_Draws" AS "Draws",
        "Fighter_2_Avg_Fight_Time_Mins" AS "Avg_Fight_Time_Mins",
        "Fighter_2_Sign_Strikes_Landed_per_Min" AS "Sign_Strikes_Landed_per_Min",
        "Fighter_2_Striking_Accuracy" AS "Striking_Accuracy",
        "Fighter_2_Sign_Strikes_Absorbed_per_Min" AS "Sign_Strikes_Absorbed_per_Min",
        "Fighter_2_Striking_Defense" AS "Striking_Defense",
        "Fighter_2_Takedown_Average_per_15Min" AS "Takedown_Average_per_15Min",
        "Fighter_2_Takedown_Accuracy" AS "Takedown_Accuracy",
        "Fighter_2_Takedown_Defense" AS "Takedown_Defense",
        "Fighter_2_Submission_Average_per_15M" AS "Submission_Average_per_15M"
    FROM
        raw_fighters_stats_before_fight
    UNION
    -- The following are the current stats after last fight
    SELECT
        NULL AS "Fight_ID",
        "ID",
        "Name",
        EXTRACT(YEAR FROM AGE(CURRENT_DATE, "Date_of_Birth")) AS "Age",
        "Wins",
        "Loses",
        "Draws",
        "Average_Fight_Time_Minutes" AS "Avg_Fight_Time_Mins",
        "Strikes_Landed_per_Minute" AS "Sign_Strikes_Landed_per_Min",
        "Striking_Accuracy" AS "Striking_Accuracy",
        "Strikes_Absorbed_per_Minute" AS "Sign_Strikes_Absorbed_per_Min",
        "Striking_Defense" AS "Striking_Defense",
        "Average_Takedowns" AS "Takedown_Average_per_15Min",
        "Takedown_Accuracy" AS "Takedown_Accuracy",
        "Takedown_Defense" AS "Takedown_Defense",
        "Submissions_Average" AS "Submission_Average_per_15M"
    FROM
        raw_fighters_current_stats
),
fighter_stats_before_fights_dates AS (
    SELECT
        fighter_stats_before_fights."ID",
        raw_fight_stats."Date" AS "Fight_Date",
        fighter_stats_before_fights."Name",
        fighter_stats_before_fights."Age",
        fighter_stats_before_fights."Wins",
        fighter_stats_before_fights."Loses",
        fighter_stats_before_fights."Draws",
        fighter_stats_before_fights."Avg_Fight_Time_Mins",
        fighter_stats_before_fights."Sign_Strikes_Landed_per_Min",
        fighter_stats_before_fights."Striking_Accuracy",
        fighter_stats_before_fights."Sign_Strikes_Absorbed_per_Min",
        fighter_stats_before_fights."Striking_Defense",
        fighter_stats_before_fights."Takedown_Average_per_15Min",
        fighter_stats_before_fights."Takedown_Accuracy",
        fighter_stats_before_fights."Takedown_Defense",
        fighter_stats_before_fights."Submission_Average_per_15M"
    FROM
        fighter_stats_before_fights
        LEFT JOIN raw_fight_stats
            ON fighter_stats_before_fights."Fight_ID" = raw_fight_stats."Fight_ID"
),
fighter_stats_get_previous_fight_date AS (
    SELECT
        *,
        LAG("Fight_Date") OVER (
            PARTITION BY "ID"
            ORDER BY "Fight_Date"
        ) AS "Previous_Fight_Date"
    FROM fighter_stats_before_fights_dates
)

SELECT
    fighter_stats_history."ID" AS "DIM_Fighter_ID",
    fighter_stats_history."Name" AS "DIM_Fighter_Name",
    fighter_stats_history."Age" AS "DIM_Fighter_Age",
    fighter_stats_history."Wins" AS "DIM_Fighter_Wins",
    fighter_stats_history."Loses" AS "DIM_Fighter_Loses",
    fighter_stats_history."Draws" AS "DIM_Fighter_Draws",
    fighter_stats_history."Avg_Fight_Time_Mins"
        AS "DIM_Fighter_Average_Fight_Time_Minutes",
    fighter_current_stats."Height_cm" AS "DIM_Fighter_Height_cm",
    fighter_current_stats."Weight_lbs" AS "DIM_Fighter_Weight_lbs",
    fighter_current_stats."Reach_cm" AS "DIM_Fighter_Reach_cm",
    fighter_current_stats."Stance" AS "DIM_Fighter_Stance",
    fighter_current_stats."Date_of_Birth" AS "DIM_Fighter_Date_of_Birth",
    fighter_stats_history."Sign_Strikes_Landed_per_Min"
        AS "DIM_Fighter_Strikes_Landed_per_Minute",
    fighter_stats_history."Striking_Accuracy"
        AS "DIM_Fighter_Striking_Accuracy",
    fighter_stats_history."Sign_Strikes_Absorbed_per_Min"
        AS "DIM_Fighter_Strikes_Absorbed_per_Minute",
    fighter_stats_history."Striking_Defense"
        AS "DIM_Fighter_Striking_Defense",
    fighter_stats_history."Takedown_Average_per_15Min"
        AS "DIM_Fighter_Average_Takedowns_per_15_Mins",
    fighter_stats_history."Takedown_Defense"
        AS "DIM_Fighter_Takedown_Defense",
    fighter_stats_history."Takedown_Accuracy"
        AS "DIM_Fighter_Takedown_Accuracy",
    fighter_stats_history."Submission_Average_per_15M"
        AS "DIM_Fighter_Submissions_Average_per_15_Mins",
    COALESCE(fighter_stats_history."Previous_Fight_Date", '1900-01-01')
        AS "DIM_Fighter_Effective_From",
    COALESCE(fighter_stats_history."Fight_Date" - INTERVAL '1 DAY', '9999-01-01')
        AS "DIM_Fighter_Effective_Until"
FROM
    fighter_stats_get_previous_fight_date fighter_stats_history
    INNER JOIN raw_fighters_current_stats fighter_current_stats
        ON fighter_stats_history."ID" = fighter_current_stats."ID"