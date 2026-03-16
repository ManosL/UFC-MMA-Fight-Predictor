INSERT INTO "FACT_Fight" (
    "FACT_Fight_ID",
    "FACT_Fight_Date_Key",
    "FACT_Fight_Gender_Key",
    "FACT_Fight_Weight_Class_Key",
    "FACT_Fight_Title_Fight",
    "FACT_Fight_Method_Key",
    "FACT_Fight_Round",
    "FACT_Fight_Time",
    "FACT_Fight_Fight_Time_Format_Key",
    "FACT_Fight_Duration_Mins"
)
SELECT
    raw_fight_stats."Fight_ID" AS "FACT_Fight_ID",
    TO_CHAR(raw_fight_stats."Date", 'YYYYMMDD') AS "FACT_Fight_Date_Key",
    dim_gender."DIM_Gender_Key" AS "FACT_Fight_Gender_Key",
    dim_weight_class."DIM_Weight_Class_Key" AS "FACT_Fight_Weight_Class_Key",
    raw_fight_stats."Title_Fight" AS "FACT_Fight_Title_Fight",
    dim_method."DIM_Method_Key" AS "FACT_Fight_Method_Key",
    raw_fight_stats."Round" AS "FACT_Fight_Round",
    raw_fight_stats."Time" AS "FACT_Fight_Time",
    dim_fight_time_format."DIM_Fight_Time_Format_Key" AS "FACT_Fight_Fight_Time_Format_Key",
    raw_fight_stats."Duration_Mins" AS "FACT_Fight_Duration_Mins"
FROM
    raw_fight_stats
    INNER JOIN "DIM_Gender" dim_gender
        ON raw_fight_stats."Gender" = dim_gender."DIM_Gender_Name"
    INNER JOIN "DIM_Weight_Class" dim_weight_class
        ON raw_fight_stats."Weight_Class" = dim_weight_class."DIM_Weight_Class_Name"
    INNER JOIN "DIM_Result" dim_result
        ON raw_fight_stats."Result" = dim_result."DIM_Result_Name"
    INNER JOIN "DIM_Method" dim_method
        ON raw_fight_stats."Method" = dim_method."DIM_Method_Name"
    INNER JOIN "DIM_Fight_Time_Format" dim_fight_time_format
        ON raw_fight_stats."Fight_Time_Format" = dim_fight_time_format."DIM_Fight_Time_Format_Name"
