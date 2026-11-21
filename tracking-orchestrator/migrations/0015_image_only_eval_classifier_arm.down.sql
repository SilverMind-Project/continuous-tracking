SET search_path = continuous_tracking, public;

DELETE FROM continuous_tracking.image_only_evaluation_daily
WHERE arm = 'classifier';

ALTER TABLE continuous_tracking.image_only_evaluation_daily
    DROP CONSTRAINT IF EXISTS image_only_evaluation_daily_arm_check;

ALTER TABLE continuous_tracking.image_only_evaluation_daily
    ADD CONSTRAINT image_only_evaluation_daily_arm_check
    CHECK (arm IN ('control', 'treatment'));
