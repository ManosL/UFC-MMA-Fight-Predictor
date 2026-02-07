SELECT 'raw_fight_stats' AS table_name, COUNT(*) AS total_rows FROM "raw_fight_stats"
UNION ALL
SELECT 'raw_fighters_current_stats', COUNT(*) FROM "raw_fighters_current_stats"
UNION ALL
SELECT 'DIM_Date', COUNT(*) FROM "DIM_Date"
UNION ALL
SELECT 'DIM_Fight_Time_Format', COUNT(*) FROM "DIM_Fight_Time_Format"
UNION ALL
SELECT 'DIM_Fighter', COUNT(*) FROM "DIM_Fighter"
UNION ALL
SELECT 'DIM_Gender', COUNT(*) FROM "DIM_Gender"
UNION ALL
SELECT 'DIM_Method', COUNT(*) FROM "DIM_Method"
UNION ALL
SELECT 'DIM_Result', COUNT(*) FROM "DIM_Result"
UNION ALL
SELECT 'DIM_Weight_Class', COUNT(*) FROM "DIM_Weight_Class"
UNION ALL
SELECT 'FACT_Fight', COUNT(*) FROM "FACT_Fight";