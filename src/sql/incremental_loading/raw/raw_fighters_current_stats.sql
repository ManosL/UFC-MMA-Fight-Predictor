MERGE INTO "raw_fighters_current_stats" AS raw_fighters_current_stats
USING "new_fighters_current_stats" AS new_fighters_current_stats
ON raw_fighters_current_stats."ID" = new_fighters_current_stats."ID"
WHEN MATCHED THEN
    UPDATE SET
        "Wins" = new_fighters_current_stats."Wins",
        "Loses" = new_fighters_current_stats."Loses",
        "Draws" = new_fighters_current_stats."Draws",
        "Average_Fight_Time_Minutes" = (
            SELECT
                COALESCE(AVG("Duration_Mins"), 0.0)
            FROM
                raw_fight_stats
            WHERE
                new_fighters_current_stats."ID" IN
                    (raw_fight_stats."Fighter_1_ID", raw_fight_stats."Fighter_2_ID")
        ),
        "Height_cm" = new_fighters_current_stats."Height_cm",
        "Weight_lbs" = new_fighters_current_stats."Weight_lbs",
        "Reach_cm" = new_fighters_current_stats."Reach_cm",
        "Stance" = new_fighters_current_stats."Stance",
        "Date_of_Birth" = new_fighters_current_stats."Date_of_Birth",
        "Total_Logged_Fights" = (
            SELECT
                COALESCE(COUNT(1), 0)
            FROM
                raw_fight_stats
            WHERE
                new_fighters_current_stats."ID" IN
                    (raw_fight_stats."Fighter_1_ID", raw_fight_stats."Fighter_2_ID")
        ),
        "Strikes_Landed_per_Minute" = new_fighters_current_stats."Strikes_Landed_per_Minute",
        "Striking_Accuracy" = new_fighters_current_stats."Striking_Accuracy",
        "Strikes_Absorbed_per_Minute" = new_fighters_current_stats."Strikes_Absorbed_per_Minute",
        "Striking_Defense" = new_fighters_current_stats."Striking_Defense",
        "Average_Takedowns" = new_fighters_current_stats."Average_Takedowns",
        "Takedown_Accuracy" = new_fighters_current_stats."Takedown_Accuracy",
        "Takedown_Defense" = new_fighters_current_stats."Takedown_Defense",
        "Submissions_Average" = new_fighters_current_stats."Submissions_Average"
WHEN NOT MATCHED THEN
    INSERT (
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
    VALUES (
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
        ),
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
        ),
        new_fighters_current_stats."Strikes_Landed_per_Minute",
        new_fighters_current_stats."Striking_Accuracy",
        new_fighters_current_stats."Strikes_Absorbed_per_Minute",
        new_fighters_current_stats."Striking_Defense",
        new_fighters_current_stats."Average_Takedowns",
        new_fighters_current_stats."Takedown_Accuracy",
        new_fighters_current_stats."Takedown_Defense",
        new_fighters_current_stats."Submissions_Average"
    );