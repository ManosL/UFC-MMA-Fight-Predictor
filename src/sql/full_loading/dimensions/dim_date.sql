INSERT INTO "DIM_Date" (
    "DIM_Date_Key",
    "DIM_Date_Full",
    "DIM_Date_Day",
    "DIM_Date_Day_Name",
    "DIM_Date_Month",
    "DIM_Date_Month_Name",
    "DIM_Date_Year",
    "DIM_Date_Day_of_Year",
    "DIM_Date_Day_of_Week",
    "DIM_Date_Quarter",
    "DIM_Date_Is_Month_Start",
    "DIM_Date_Is_Month_End",
    "DIM_Date_Is_Quarter_Start",
    "DIM_Date_Is_Quarter_End",
    "DIM_Date_Is_Year_Start",
    "DIM_Date_Is_Year_End",
    "DIM_Date_Is_Leap_Year"
)
SELECT
    TO_CHAR(d, 'YYYYMMDD') AS "DIM_Date_Key",
    d AS "DIM_Date_Full",
    EXTRACT(DAY FROM d)::SMALLINT AS "DIM_Date_Day",
    TO_CHAR(d, 'Day') AS "DIM_Date_Day_Name",
    EXTRACT(MONTH FROM d)::SMALLINT AS "DIM_Date_Month",
    TO_CHAR(d, 'Month') AS "DIM_Date_Month_Name",
    EXTRACT(YEAR FROM d)::SMALLINT AS "DIM_Date_Year",
    EXTRACT(DOY FROM d)::SMALLINT AS "DIM_Date_Day_of_Year",
    EXTRACT(ISODOW FROM d)::SMALLINT AS "DIM_Date_Day_of_Week",
    EXTRACT(QUARTER FROM d)::SMALLINT AS "DIM_Date_Quarter",
    (d = DATE_TRUNC('month', d)) AS "DIM_Date_Is_Month_Start",
    (d = (DATE_TRUNC('month', d) + INTERVAL '1 MONTH - 1 DAY')::DATE) AS "DIM_Date_Is_Month_End",
    (d = DATE_TRUNC('quarter', d)) AS "DIM_Date_Is_Quarter_Start",
    (d = (DATE_TRUNC('quarter', d) + INTERVAL '3 MONTH - 1 DAY')::DATE) AS "DIM_Date_Is_Quarter_End",
    (d = DATE_TRUNC('year', d)) AS "DIM_Date_Is_Year_Start",
    (d = (DATE_TRUNC('year', d) + INTERVAL '1 YEAR - 1 DAY')::DATE) AS "DIM_Date_Is_Year_End",
    EXTRACT(YEAR FROM d)::INT % 4 = 0
        AND (EXTRACT(YEAR FROM d)::INT % 100 <> 0 OR EXTRACT(YEAR FROM d)::INT % 400 = 0)
        AS "DIM_Date_Is_Leap_Year"
FROM
    GENERATE_SERIES(
        '1970-01-01'::date,
        '2050-12-31'::date,
        INTERVAL '1 DAY'
    ) AS t(d);
