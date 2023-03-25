DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOOP_SCRIPT=$DIR/0.4_korba3_loop.sh

$LOOP_SCRIPT "$DIR/1.0_merge.sh"
$LOOP_SCRIPT "$DIR/1.1_assert_merge.sh"
$LOOP_SCRIPT "$DIR/1.2_baselines.sh"
$LOOP_SCRIPT "$DIR/1.3_stats.sh"

#CV
export CVN=5 # number of cross validation folds
#$LOOP_SCRIPT "$DIR/1.4_cv.sh"
$DIR/1.4_cv.sh korba3-korba_reczna


CV_LOOP_SCRIPT=$DIR/0.5_korba3_cv_loop.sh

#wydzielić dev z train
$CV_LOOP_SCRIPT "$DIR/1.6_cv_split.sh"

# polaczyc trainy z innymi z shufflem

# przygotować dane do treningu tokenizera i tagera

#train
$CV_LOOP_SCRIPT "$DIR/1.7_train_tokenizer.sh"

$CV_LOOP_SCRIPT "$DIR/1.8_test_tokenizer.sh"

#only korba, korba + f19, korba+nkjp, kobra+f19+nkjp, dev and test only korba
#czy dla każdego splitu dodawać cały korpus dodatkowy zamiast splitu? - co z zbalansowaniem - dotrenowanie?

$CV_LOOP_SCRIPT "$DIR/3.0_prepare_tagger.sh"
$CV_LOOP_SCRIPT "$DIR/3.2_train_tagger.sh"

$CV_LOOP_SCRIPT "$DIR/3.3_test_tagger.sh"


#scripts/4.0_merge_datasets.sh
$CV_LOOP_SCRIPT "$DIR/4.1_train_tokenizer.sh"
$CV_LOOP_SCRIPT "$DIR/4.1_train_tagger.sh"