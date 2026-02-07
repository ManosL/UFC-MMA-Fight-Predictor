MERGE INTO "DIM_Weight_Class" AS dim_weight_class
USING (
    SELECT
        DISTINCT "Weight_Class"
    FROM
        raw_fight_stats
    ORDER BY
        "Weight_Class"
) AS raw_weight_class
ON
    dim_weight_class."DIM_Weight_Class_Name" = raw_weight_class."Weight_Class"
-- Do nothing when match because it's pretty unlikely the fight stats to change
WHEN NOT MATCHED THEN
    INSERT (
        "DIM_Weight_Class_Name"
    )
    VALUES (
       raw_weight_class."Weight_Class"
);
