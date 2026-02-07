INSERT INTO "raw_fighters_current_stats"(
        "ID",
        "Gender",
        "Name",
        "Wins",
        "Loses",
        "Draws",
        "Average_Fight_Time_Minutes",
        "Height_cm",
        "Weight_lbs",
        "Reach_cm",
        "Stance",
        "Date_of_Birth",
        "Total_Logged_Fights",
        "Strikes_Landed_per_Minute",
        "Striking_Accuracy",
        "Strikes_Absorbed_per_Minute",
        "Striking_Defense",
        "Average_Takedowns",
        "Takedown_Accuracy",
        "Takedown_Defense",
        "Submissions_Average"
    )
SELECT
    new_fighters_current_stats."ID",
    new_fighters_current_stats."Gender",
    new_fighters_current_stats."Name",
    new_fighters_current_stats."Wins",
    new_fighters_current_stats."Loses",
    new_fighters_current_stats."Draws",
    (
        SELECT
            COALESCE(AVG("Duration_Mins"), 0.0)
        FROM
            raw_fight_stats
        WHERE
            new_fighters_current_stats."ID" IN
                (raw_fight_stats."Fighter_1_ID", raw_fight_stats."Fighter_2_ID")
    ) AS "Average_Fight_Time_Minutes",
    new_fighters_current_stats."Height_cm",
    new_fighters_current_stats."Weight_lbs",
    new_fighters_current_stats."Reach_cm",
    new_fighters_current_stats."Stance",
    new_fighters_current_stats."Date_of_Birth",
    (
        SELECT
            COALESCE(COUNT(1), 0)
        FROM
            raw_fight_stats
        WHERE
            new_fighters_current_stats."ID" IN
                (raw_fight_stats."Fighter_1_ID", raw_fight_stats."Fighter_2_ID")
    ) AS "Total_Logged_Fights",
    new_fighters_current_stats."Strikes_Landed_per_Minute",
    new_fighters_current_stats."Striking_Accuracy",
    new_fighters_current_stats."Strikes_Absorbed_per_Minute",
    new_fighters_current_stats."Striking_Defense",
    new_fighters_current_stats."Average_Takedowns",
    new_fighters_current_stats."Takedown_Accuracy",
    new_fighters_current_stats."Takedown_Defense",
    new_fighters_current_stats."Submissions_Average"
FROM
    new_fighters_current_stats;