INSERT INTO "FACT_Fight_Fighter_Stats" (
    "FACT_Fight_Fighter_Stats_Fight_Key",
    "FACT_Fight_Fighter_Stats_Result_Key",
    "FACT_Fight_Fighter_Stats_Fighter_Key",
    "FACT_Fight_Fighter_Stats_Fighter_Corner",
    "FACT_Fight_Fighter_Stats_Knock_Downs",
    "FACT_Fight_Fighter_Stats_Sign.Strikes_Done",
    "FACT_Fight_Fighter_Stats_Sign.Strikes_Attempted",
    "FACT_Fight_Fighter_Stats_Sign.Strikes_Perc.",
    "FACT_Fight_Fighter_Stats_Total_Strikes_Done",
    "FACT_Fight_Fighter_Stats_Total_Strikes_Attempted",
    "FACT_Fight_Fighter_Stats_Takedowns_Done",
    "FACT_Fight_Fighter_Stats_Takedowns_Attempted",
    "FACT_Fight_Fighter_Stats_Takedowns_Perc.",
    "FACT_Fight_Fighter_Stats_Submission_Attempts",
    "FACT_Fight_Fighter_Stats_Rev",
    "FACT_Fight_Fighter_Stats_Control",
    "FACT_Fight_Fighter_Stats_Opponent_Knock_Downs",
    "FACT_Fight_Fighter_Stats_Opponent_Sign.Strikes_Done",
    "FACT_Fight_Fighter_Stats_Opponent_Sign.Strikes_Attempted",
    "FACT_Fight_Fighter_Stats_Opponent_Sign.Strikes_Perc.",
    "FACT_Fight_Fighter_Stats_Opponent_Total_Strikes_Done",
    "FACT_Fight_Fighter_Stats_Opponent_Total_Strikes_Attempted",
    "FACT_Fight_Fighter_Stats_Opponent_Takedowns_Done",
    "FACT_Fight_Fighter_Stats_Opponent_Takedowns_Attempted",
    "FACT_Fight_Fighter_Stats_Opponent_Takedowns_Perc.",
    "FACT_Fight_Fighter_Stats_Opponent_Submission_Attempts",
    "FACT_Fight_Fighter_Stats_Opponent_Rev",
    "FACT_Fight_Fighter_Stats_Opponent_Control"
)
WITH corner_1 AS (
    SELECT
        fact_fight."FACT_Fight_Key" AS "FACT_Fight_Fighter_Stats_Fight_Key",
        dim_result."DIM_Result_Key" AS "FACT_Fight_Fighter_Stats_Result_Key",
        dim_fighter_1."DIM_Fighter_Key" AS "FACT_Fight_Fighter_Stats_Fighter_Key",
        1 AS "FACT_Fight_Fighter_Stats_Fighter_Corner",
        raw_fight_stats."Fighter_1_Knock_Downs" AS "FACT_Fight_Fighter_Stats_Knock_Downs",
        raw_fight_stats."Fighter_1_Sign.Strikes_Done" AS "FACT_Fight_Fighter_Stats_Sign.Strikes_Done",
        raw_fight_stats."Fighter_1_Sign.Strikes_Attempted" AS "FACT_Fight_Fighter_Stats_Sign.Strikes_Attempted",
        raw_fight_stats."Fighter_1_Sign.Strikes_Perc." AS "FACT_Fight_Fighter_Stats_Sign.Strikes_Perc.",
        raw_fight_stats."Fighter_1_Total_Strikes_Done" AS "FACT_Fight_Fighter_Stats_Total_Strikes_Done",
        raw_fight_stats."Fighter_1_Total_Strikes_Attempted" AS "FACT_Fight_Fighter_Stats_Total_Strikes_Attempted",
        raw_fight_stats."Fighter_1_Takedowns_Done" AS "FACT_Fight_Fighter_Stats_Takedowns_Done",
        raw_fight_stats."Fighter_1_Takedowns_Attempted" AS "FACT_Fight_Fighter_Stats_Takedowns_Attempted",
        raw_fight_stats."Fighter_1_Takedowns_Perc." AS "FACT_Fight_Fighter_Stats_Takedowns_Perc.",
        raw_fight_stats."Fighter_1_Submission_Attempts" AS "FACT_Fight_Fighter_Stats_Submission_Attempts",
        raw_fight_stats."Fighter_1_Rev" AS "FACT_Fight_Fighter_Stats_Rev",
        raw_fight_stats."Fighter_1_Control" AS "FACT_Fight_Fighter_Stats_Control",
        raw_fight_stats."Fighter_2_Knock_Downs" AS "FACT_Fight_Fighter_Stats_Opponent_Knock_Downs",
        raw_fight_stats."Fighter_2_Sign.Strikes_Done" AS "FACT_Fight_Fighter_Stats_Opponent_Sign.Strikes_Done",
        raw_fight_stats."Fighter_2_Sign.Strikes_Attempted" AS "FACT_Fight_Fighter_Stats_Opponent_Sign.Strikes_Attempted",
        raw_fight_stats."Fighter_2_Sign.Strikes_Perc." AS "FACT_Fight_Fighter_Stats_Opponent_Sign.Strikes_Perc.",
        raw_fight_stats."Fighter_2_Total_Strikes_Done" AS "FACT_Fight_Fighter_Stats_Opponent_Total_Strikes_Done",
        raw_fight_stats."Fighter_2_Total_Strikes_Attempted" AS "FACT_Fight_Fighter_Stats_Opponent_Total_Strikes_Attempted",
        raw_fight_stats."Fighter_2_Takedowns_Done" AS "FACT_Fight_Fighter_Stats_Opponent_Takedowns_Done",
        raw_fight_stats."Fighter_2_Takedowns_Attempted" AS "FACT_Fight_Fighter_Stats_Opponent_Takedowns_Attempted",
        raw_fight_stats."Fighter_2_Takedowns_Perc." AS "FACT_Fight_Fighter_Stats_Opponent_Takedowns_Perc.",
        raw_fight_stats."Fighter_2_Submission_Attempts" AS "FACT_Fight_Fighter_Stats_Opponent_Submission_Attempts",
        raw_fight_stats."Fighter_2_Rev" AS "FACT_Fight_Fighter_Stats_Opponent_Rev",
        raw_fight_stats."Fighter_2_Control" AS "FACT_Fight_Fighter_Stats_Opponent_Control"
    FROM
        raw_fight_stats
        INNER JOIN "FACT_Fight" fact_fight
            ON raw_fight_stats."Fight_ID" = fact_fight."FACT_Fight_ID"
        INNER JOIN "DIM_Result" dim_result
            ON raw_fight_stats."Result" = dim_result."DIM_Result_Name"
        INNER JOIN "DIM_Fighter" dim_fighter_1
            ON raw_fight_stats."Fighter_1_ID" = dim_fighter_1."DIM_Fighter_ID" AND
            dim_fighter_1."DIM_Fighter_Effective_Until" = raw_fight_stats."Date" - INTERVAL '1 DAY'
        INNER JOIN "DIM_Fighter" dim_fighter_2
            ON raw_fight_stats."Fighter_2_ID" = dim_fighter_2."DIM_Fighter_ID" AND
            dim_fighter_2."DIM_Fighter_Effective_Until" = raw_fight_stats."Date" - INTERVAL '1 DAY'
),
corner_2 AS (
    SELECT
        fact_fight."FACT_Fight_Key" AS "FACT_Fight_Fighter_Stats_Fight_Key",
        dim_result."DIM_Result_Key" AS "FACT_Fight_Fighter_Stats_Result_Key",
        dim_fighter_1."DIM_Fighter_Key" AS "FACT_Fight_Fighter_Stats_Fighter_Key",
        2 AS "FACT_Fight_Fighter_Stats_Fighter_Corner",
        raw_fight_stats."Fighter_2_Knock_Downs" AS "FACT_Fight_Fighter_Stats_Knock_Downs",
        raw_fight_stats."Fighter_2_Sign.Strikes_Done" AS "FACT_Fight_Fighter_Stats_Sign.Strikes_Done",
        raw_fight_stats."Fighter_2_Sign.Strikes_Attempted" AS "FACT_Fight_Fighter_Stats_Sign.Strikes_Attempted",
        raw_fight_stats."Fighter_2_Sign.Strikes_Perc." AS "FACT_Fight_Fighter_Stats_Sign.Strikes_Perc.",
        raw_fight_stats."Fighter_2_Total_Strikes_Done" AS "FACT_Fight_Fighter_Stats_Total_Strikes_Done",
        raw_fight_stats."Fighter_2_Total_Strikes_Attempted" AS "FACT_Fight_Fighter_Stats_Total_Strikes_Attempted",
        raw_fight_stats."Fighter_2_Takedowns_Done" AS "FACT_Fight_Fighter_Stats_Takedowns_Done",
        raw_fight_stats."Fighter_2_Takedowns_Attempted" AS "FACT_Fight_Fighter_Stats_Takedowns_Attempted",
        raw_fight_stats."Fighter_2_Takedowns_Perc." AS "FACT_Fight_Fighter_Stats_Takedowns_Perc.",
        raw_fight_stats."Fighter_2_Submission_Attempts" AS "FACT_Fight_Fighter_Stats_Submission_Attempts",
        raw_fight_stats."Fighter_2_Rev" AS "FACT_Fight_Fighter_Stats_Rev",
        raw_fight_stats."Fighter_2_Control" AS "FACT_Fight_Fighter_Stats_Control",
        raw_fight_stats."Fighter_1_Knock_Downs" AS "FACT_Fight_Fighter_Stats_Opponent_Knock_Downs",
        raw_fight_stats."Fighter_1_Sign.Strikes_Done" AS "FACT_Fight_Fighter_Stats_Opponent_Sign.Strikes_Done",
        raw_fight_stats."Fighter_1_Sign.Strikes_Attempted" AS "FACT_Fight_Fighter_Stats_Opponent_Sign.Strikes_Attempted",
        raw_fight_stats."Fighter_1_Sign.Strikes_Perc." AS "FACT_Fight_Fighter_Stats_Opponent_Sign.Strikes_Perc.",
        raw_fight_stats."Fighter_1_Total_Strikes_Done" AS "FACT_Fight_Fighter_Stats_Opponent_Total_Strikes_Done",
        raw_fight_stats."Fighter_1_Total_Strikes_Attempted" AS "FACT_Fight_Fighter_Stats_Opponent_Total_Strikes_Attempted",
        raw_fight_stats."Fighter_1_Takedowns_Done" AS "FACT_Fight_Fighter_Stats_Opponent_Takedowns_Done",
        raw_fight_stats."Fighter_1_Takedowns_Attempted" AS "FACT_Fight_Fighter_Stats_Opponent_Takedowns_Attempted",
        raw_fight_stats."Fighter_1_Takedowns_Perc." AS "FACT_Fight_Fighter_Stats_Opponent_Takedowns_Perc.",
        raw_fight_stats."Fighter_1_Submission_Attempts" AS "FACT_Fight_Fighter_Stats_Opponent_Submission_Attempts",
        raw_fight_stats."Fighter_1_Rev" AS "FACT_Fight_Fighter_Stats_Opponent_Rev",
        raw_fight_stats."Fighter_1_Control" AS "FACT_Fight_Fighter_Stats_Opponent_Control"
    FROM
        raw_fight_stats
        INNER JOIN "FACT_Fight" fact_fight
            ON raw_fight_stats."Fight_ID" = fact_fight."FACT_Fight_ID"
        INNER JOIN "DIM_Result" dim_result
            ON dim_result."DIM_Result_Name" =
                CASE
                    WHEN raw_fight_stats."Result" = 'win' THEN 'lose'
                    WHEN raw_fight_stats."Result" = 'lose' THEN 'win'
                    ELSE raw_fight_stats."Result"
                END
        INNER JOIN "DIM_Fighter" dim_fighter_1
            ON raw_fight_stats."Fighter_1_ID" = dim_fighter_1."DIM_Fighter_ID" AND
            dim_fighter_1."DIM_Fighter_Effective_Until" = raw_fight_stats."Date" - INTERVAL '1 DAY'
        INNER JOIN "DIM_Fighter" dim_fighter_2
            ON raw_fight_stats."Fighter_2_ID" = dim_fighter_2."DIM_Fighter_ID" AND
            dim_fighter_2."DIM_Fighter_Effective_Until" = raw_fight_stats."Date" - INTERVAL '1 DAY'
)

SELECT * FROM corner_1
UNION
SELECT * FROM corner_2
