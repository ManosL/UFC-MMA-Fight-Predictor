DROP VIEW IF EXISTS "ML_Fights";

CREATE VIEW "ML_Fights" AS
SELECT
    fact_fight."FACT_Fight_Key" AS "Fight_ID",
    dim_date."DIM_Date_Full" AS "Fight_Date",
    dim_gender."DIM_Gender_Name" AS "Gender",
    dim_weight_class."DIM_Weight_Class_Name" AS "Weight_Class",
    fact_fight."FACT_Fight_Title_Fight" AS "Title_Fight",
    dim_result."DIM_Result_Name" AS "Result",
    dim_method."DIM_Method_Name" AS "Method",
    fact_fight."FACT_Fight_Round" AS "Round",
    fact_fight."FACT_Fight_Time" AS "Time",
    dim_fight_time_format."DIM_Fight_Time_Format_Name" AS "Fight_Time_Format",
    dim_fighter_1."DIM_Fighter_ID" AS "Fighter_1_ID",
    dim_fighter_1."DIM_Fighter_Name" AS "Fighter_1_Name",
    DATE_PART(
        'year',
        AGE(CURRENT_DATE, dim_fighter_1."DIM_Fighter_Date_of_Birth")
    ) AS "Fighter_1_Age",
    dim_fighter_1."DIM_Fighter_Wins" AS "Fighter_1_Wins",
    dim_fighter_1."DIM_Fighter_Loses" AS "Fighter_1_Loses",
    dim_fighter_1."DIM_Fighter_Draws" AS "Fighter_1_Draws",
    dim_fighter_1."DIM_Fighter_Average_Fight_Time_Minutes" AS "Fighter_1_Avg_Time(MINS)",
    dim_fighter_1."DIM_Fighter_Height_cm" AS "Fighter_1_Height",
    dim_fighter_1."DIM_Fighter_Reach_cm" AS "Fighter_1_Reach",
    dim_fighter_1."DIM_Fighter_Stance" AS "Fighter_1_Stance",
    dim_fighter_1."DIM_Fighter_Strikes_Landed_per_Minute" AS "Fighter_1_Sign_SLpMin",
    dim_fighter_1."DIM_Fighter_Striking_Accuracy" AS "Fighter_1_Str_Acc",
    dim_fighter_1."DIM_Fighter_Strikes_Absorbed_per_Minute" AS "Fighter_1_Sign_SApMin",
    dim_fighter_1."DIM_Fighter_Striking_Defense" AS "Fighter_1_Defense",
    dim_fighter_1."DIM_Fighter_Average_Takedowns_per_15_Mins" AS "Fighter_1_Takedown_Avgp15M",
    dim_fighter_1."DIM_Fighter_Takedown_Accuracy" AS "Fighter_1_Takedown_Acc",
    dim_fighter_1."DIM_Fighter_Takedown_Defense" AS "Fighter_1_Takedown_Def",
    dim_fighter_1."DIM_Fighter_Submissions_Average_per_15_Mins" AS "Fighter_1_Sub_Avgp15M",
    dim_fighter_2."DIM_Fighter_ID" AS "Fighter_2_ID",
    dim_fighter_2."DIM_Fighter_Name" AS "Fighter_2_Name",
    DATE_PART(
        'year',
        AGE(CURRENT_DATE, dim_fighter_2."DIM_Fighter_Date_of_Birth")
    ) AS "Fighter_2_Age",
    dim_fighter_2."DIM_Fighter_Wins" AS "Fighter_2_Wins",
    dim_fighter_2."DIM_Fighter_Loses" AS "Fighter_2_Loses",
    dim_fighter_2."DIM_Fighter_Draws" AS "Fighter_2_Draws",
    dim_fighter_2."DIM_Fighter_Average_Fight_Time_Minutes" AS "Fighter_2_Avg_Time(MINS)",
    dim_fighter_2."DIM_Fighter_Height_cm" AS "Fighter_2_Height",
    dim_fighter_2."DIM_Fighter_Reach_cm" AS "Fighter_2_Reach",
    dim_fighter_2."DIM_Fighter_Stance" AS "Fighter_2_Stance",
    dim_fighter_2."DIM_Fighter_Strikes_Landed_per_Minute" AS "Fighter_2_Sign_SLpMin",
    dim_fighter_2."DIM_Fighter_Striking_Accuracy" AS "Fighter_2_Str_Acc",
    dim_fighter_2."DIM_Fighter_Strikes_Absorbed_per_Minute" AS "Fighter_2_Sign_SApMin",
    dim_fighter_2."DIM_Fighter_Striking_Defense" AS "Fighter_2_Defense",
    dim_fighter_2."DIM_Fighter_Average_Takedowns_per_15_Mins" AS "Fighter_2_Takedown_Avgp15M",
    dim_fighter_2."DIM_Fighter_Takedown_Accuracy" AS "Fighter_2_Takedown_Acc",
    dim_fighter_2."DIM_Fighter_Takedown_Defense" AS "Fighter_2_Takedown_Def",
    dim_fighter_2."DIM_Fighter_Submissions_Average_per_15_Mins" AS "Fighter_2_Sub_Avgp15M"
FROM
    "FACT_Fight" fact_fight
    INNER JOIN "FACT_Fight_Fighter_Stats" fact_fight_fighter_stats_1
        ON fact_fight."FACT_Fight_Key" = fact_fight_fighter_stats_1."FACT_Fight_Fighter_Stats_Fight_Key"
        AND fact_fight_fighter_stats_1."FACT_Fight_Fighter_Stats_Fighter_Corner" = 1
    INNER JOIN "FACT_Fight_Fighter_Stats" fact_fight_fighter_stats_2
        ON fact_fight."FACT_Fight_Key" = fact_fight_fighter_stats_2."FACT_Fight_Fighter_Stats_Fight_Key"
        AND fact_fight_fighter_stats_2."FACT_Fight_Fighter_Stats_Fighter_Corner" = 2
    INNER JOIN "DIM_Date" dim_date
        ON fact_fight."FACT_Fight_Date_Key" = dim_date."DIM_Date_Key"
    INNER JOIN "DIM_Gender" dim_gender
        ON fact_fight."FACT_Fight_Gender_Key" = dim_gender."DIM_Gender_Key"
    INNER JOIN "DIM_Weight_Class" dim_weight_class
        ON fact_fight."FACT_Fight_Weight_Class_Key" = dim_weight_class."DIM_Weight_Class_Key"
    INNER JOIN "DIM_Result" dim_result
        ON fact_fight_fighter_stats_1."FACT_Fight_Fighter_Stats_Result_Key" = dim_result."DIM_Result_Key"
    INNER JOIN "DIM_Method" dim_method
        ON fact_fight."FACT_Fight_Method_Key" = dim_method."DIM_Method_Key"
    INNER JOIN "DIM_Fight_Time_Format" dim_fight_time_format
        ON fact_fight."FACT_Fight_Fight_Time_Format_Key" = dim_fight_time_format."DIM_Fight_Time_Format_Key"
    INNER JOIN "DIM_Fighter" dim_fighter_1
        ON fact_fight_fighter_stats_1."FACT_Fight_Fighter_Stats_Fighter_Key" = dim_fighter_1."DIM_Fighter_Key"
    INNER JOIN "DIM_Fighter" dim_fighter_2
        ON fact_fight_fighter_stats_2."FACT_Fight_Fighter_Stats_Fighter_Key" = dim_fighter_2."DIM_Fighter_Key";
