-- Do nothing when match because it's pretty unlikely the fight stats to change
MERGE INTO "raw_fight_stats" AS raw_fight_stats
USING (
    SELECT
        *
    FROM
        "new_fight_stats"
    ORDER BY
        new_fight_stats."Date" ASC
) AS new_fight_stats
ON
    raw_fight_stats."Date" = new_fight_stats."Date" AND
    (
        (
            raw_fight_stats."Fighter_1_ID" = new_fight_stats."Fighter_1_ID" AND
            raw_fight_stats."Fighter_2_ID" = new_fight_stats."Fighter_2_ID"
        ) OR
        (
            raw_fight_stats."Fighter_1_ID" = new_fight_stats."Fighter_2_ID" AND
            raw_fight_stats."Fighter_2_ID" = new_fight_stats."Fighter_1_ID"
        )
    )
WHEN NOT MATCHED THEN
    INSERT (
        "Fight_ID",
        "Date",
        "Gender",
        "Weight_Class",
        "Title_Fight",
        "Result",
        "Method",
        "Round",
        "Time",
        "Fight_Time_Format",
        "Fighter_1_ID",
        "Fighter_1_Name",
        "Fighter_1_Nickname",
        "Fighter_1_Knock_Downs",
        "Fighter_1_Sign.Strikes_Done",
        "Fighter_1_Sign.Strikes_Attempted",
        "Fighter_1_Sign.Strikes_Perc.",
        "Fighter_1_Total_Strikes_Done",
        "Fighter_1_Total_Strikes_Attempted",
        "Fighter_1_Takedowns_Done",
        "Fighter_1_Takedowns_Attempted",
        "Fighter_1_Takedowns_Perc.",
        "Fighter_1_Submission_Attempts",
        "Fighter_1_Rev",
        "Fighter_1_Control",
        "Fighter_2_ID",
        "Fighter_2_Name",
        "Fighter_2_Nickname",
        "Fighter_2_Knock_Downs",
        "Fighter_2_Sign.Strikes_Done",
        "Fighter_2_Sign.Strikes_Attempted",
        "Fighter_2_Sign.Strikes_Perc.",
        "Fighter_2_Total_Strikes_Done",
        "Fighter_2_Total_Strikes_Attempted",
        "Fighter_2_Takedowns_Done",
        "Fighter_2_Takedowns_Attempted",
        "Fighter_2_Takedowns_Perc.",
        "Fighter_2_Submission_Attempts",
        "Fighter_2_Rev",
        "Fighter_2_Control",
        "Duration_Mins"
    )
    VALUES (
        new_fight_stats."Fight_ID",
        new_fight_stats."Date",
        new_fight_stats."Gender",
        new_fight_stats."Weight_Class",
        new_fight_stats."Title_Fight",
        new_fight_stats."Result",
        new_fight_stats."Method",
        new_fight_stats."Round",
        new_fight_stats."Time",
        new_fight_stats."Fight_Time_Format",
        new_fight_stats."Fighter_1_ID",
        new_fight_stats."Fighter_1_Name",
        new_fight_stats."Fighter_1_Nickname",
        new_fight_stats."Fighter_1_Knock_Downs",
        new_fight_stats."Fighter_1_Sign.Strikes_Done",
        new_fight_stats."Fighter_1_Sign.Strikes_Attempted",
        new_fight_stats."Fighter_1_Sign.Strikes_Perc.",
        new_fight_stats."Fighter_1_Total_Strikes_Done",
        new_fight_stats."Fighter_1_Total_Strikes_Attempted",
        new_fight_stats."Fighter_1_Takedowns_Done",
        new_fight_stats."Fighter_1_Takedowns_Attempted",
        new_fight_stats."Fighter_1_Takedowns_Perc.",
        new_fight_stats."Fighter_1_Submission_Attempts",
        new_fight_stats."Fighter_1_Rev",
        new_fight_stats."Fighter_1_Control",
        new_fight_stats."Fighter_2_ID",
        new_fight_stats."Fighter_2_Name",
        new_fight_stats."Fighter_2_Nickname",
        new_fight_stats."Fighter_2_Knock_Downs",
        new_fight_stats."Fighter_2_Sign.Strikes_Done",
        new_fight_stats."Fighter_2_Sign.Strikes_Attempted",
        new_fight_stats."Fighter_2_Sign.Strikes_Perc.",
        new_fight_stats."Fighter_2_Total_Strikes_Done",
        new_fight_stats."Fighter_2_Total_Strikes_Attempted",
        new_fight_stats."Fighter_2_Takedowns_Done",
        new_fight_stats."Fighter_2_Takedowns_Attempted",
        new_fight_stats."Fighter_2_Takedowns_Perc.",
        new_fight_stats."Fighter_2_Submission_Attempts",
        new_fight_stats."Fighter_2_Rev",
        new_fight_stats."Fighter_2_Control",
        new_fight_stats."Duration_Mins"
    );
