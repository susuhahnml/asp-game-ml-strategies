R=`tput setaf 1`
G=`tput setaf 2`
Y=`tput setaf 3`
B=`tput setaf 4`
NC=`tput sgr0`
set -e
echo "$Y Running all benchmarks for Dominnoes...$NC"

echo "$Y    ----------------$NC"
echo "$Y    | Building DQN $NC"
echo "$Y    ----------------$NC"


echo "$Y DQN against random...$NC"

python main.py dqn_rl --game-name=dom --rand=2 --out="dqn-random" --train-rand=2 --nb-steps=1000000 --save-every=50000 --model-name=random

echo "$Y DQN against strategy...$NC"

python main.py dqn_rl --game-name=dom --rand=2 --out="dqn-strategy" --train-rand=2 --nb-steps=1000000 --save-every=50000 --strategy-opponent='./approaches/dqn_rl/dom-asp-strategy.lp' --model-name=strategy 


echo ""
echo ""
echo ""
echo "$Y    -----------------------$NC"
echo "$Y    | Building Supervised $NC"
echo "$Y    -----------------------$NC"

echo "$Y Supervised ...$NC"

python main.py supervised_ml --game-name=dom --rand=2 --out="supervised" --n-epochs=5000 --model-name=transfer --training-file="corrected-200games-1000-iter.csv"

echo ""
echo ""
echo ""
echo "$Y    ----------------------$NC"
echo "$Y    | Building AlphaZero $NC"
echo "$Y    ----------------------$NC"

echo "$Y Alpha zero not penalized ...$NC"

python main.py alpha_zero --game-name=dom --rand=2 --out="alpha-not-penalized" --train-rand=2  --model-name=not-penalized --n-vs=200 --n-train=150 --n-episodes=100 --n-epochs=500 --n-mcts-simulations=500

echo "$Y Alpha zero penalized ...$NC"

python main.py alpha_zero --game-name=dom --rand=2 --out="alpha-penalized" --train-rand=2  --model-name=penalized --n-vs=200 --n-train=150 --n-episodes=100 --n-epochs=500 --n-mcts-simulations=500 --penalize-illegal

echo ""
echo ""
echo ""
echo "$Y    ---------------------$NC"
echo "$Y    | VS against random $NC"
echo "$Y    ---------------------$NC"

echo "$Y ------ VS Dqn $f..."

FILES="./approaches/dqn_rl/saved_models/dom/random/*.json"
for f in $FILES
do
    echo "$Y VS against random for $f..."
    python main.py vs --game-name=dom --rand=2 --n-initial=10 --play-symmetry --n=250 --a=dqn_rl-random/$(basename -- $f .json) --out="dqn-random/$(basename -- $f).json" --penalize-illegal
done


FILES="./approaches/dqn_rl/saved_models/dom/strategy/*.json"
for f in $FILES
do
    echo "$Y VS against random for $f... $NC"
    python main.py vs --game-name=dom --rand=2 --n-initial=10 --play-symmetry --n=250 --a=dqn_rl-strategy/$(basename -- $f .json) --out="dqn-strategy/$(basename -- $f).json" --penalize-illegal
done


echo "$Y ------ VS Supervised $f... $NC"
python main.py vs --game-name=dom --rand=2 --n-initial=10 --play-symmetry --n=250 --a=supervised_ml-transfer --out="supervised-transfer/transfer.json" --penalize-illegal


echo "$Y ------ VS Alpha zero $f... $NC"
FILES="./approaches/alpha_zero/saved_models/dom/not-penalized/*.json"
for f in $FILES
do
    echo "$Y VS against random for $f... $NC"
    python main.py vs --game-name=dom --rand=2 --n-initial=10 --play-symmetry --n=250 --a=alpha_zero-not-penalized/$(basename -- $f .json) --out="alpha-zero-not-penalized/$(basename -- $f).json" --penalize-illegal
done


FILES="./approaches/alpha_zero/saved_models/dom/penalized/*.json"
for f in $FILES
do
    echo "$Y VS against random for $f... $NC"
    python main.py vs --game-name=dom --rand=2 --n-initial=10 --play-symmetry --n=250 --a=alpha_zero-penalized/$(basename -- $f .json) --out="alpha-zero-penalized/$(basename -- $f).json" --penalize-illegal
done


echo ""
echo ""
echo ""
echo "$Y    ---------------------$NC"
echo "$Y    | Plotting results $NC"
echo "$Y    ---------------------$NC"

python main.py plot --game-name=dom --file="dqn-random" --plot-out="dqn-random"
python main.py plot --game-name=dom --file="dqn-strategy" --plot-out="dqn-strategy"

python main.py plot --game-name=dom --file="supervised-transfer" --plot-out="supervised-transfer"

python main.py plot --game-name=dom --file="supervised-transfer" --plot-out="supervised-transfer"

python main.py plot --game-name=dom --file="alpha-zero-not-penalized" --plot-out="alpha-zero-not-penalized"
python main.py plot --game-name=dom --file="alpha-zero-penalized" --plot-out="alpha-zero-penalized"


echo "$B    ------------------------------------------ $NC"
echo "$B    | Benchmarks for nim with one intial state $NC"
echo "$B    ------------------------------------------ $NC"







####################################################################################


echo "$B ------------------------------------------------------------------$NC"
echo "$B Running all benchmarks for Nim...$NC"

echo "$B    ----------------$NC"
echo "$B    | Building DQN $NC"
echo "$B    ----------------$NC"


echo "$B DQN against random...$NC"

python main.py dqn_rl --game-name=nim --initial=default_initial.lp --out="dqn-random"  --nb-steps=1000000 --save-every=50000 --model-name=random



echo ""
echo ""
echo ""
echo "$B    ----------------------$NC"
echo "$B    | Building AlphaZero $NC"
echo "$B    ----------------------$NC"

echo "$B Alpha zero not penalized ...$NC"

python main.py alpha_zero --game-name=nim --initial=default_initial.lp --out="alpha-not-penalized"   --model-name=not-penalized --n-vs=200 --n-train=150 --n-episodes=100 --n-epochs=500 --n-mcts-simulations=300

echo "$B Alpha zero penalized ...$NC"

python main.py alpha_zero --game-name=nim --initial=default_initial.lp --out="alpha-penalized"   --model-name=penalized --n-vs=200 --n-train=150 --n-episodes=100 --n-epochs=500 --n-mcts-simulations=300 --penalize-illegal

echo ""
echo ""
echo ""
echo "$B    ---------------------$NC"
echo "$B    | VS against random $NC"
echo "$B    ---------------------$NC"

echo "$B ------ VS Dqn $f..."

FILES="./approaches/dqn_rl/saved_models/nim/random/*.json"
for f in $FILES
do
    echo "$B VS against random for $f..."
    python main.py vs --game-name=nim --initial=default_initial.lp --n-initial=10 --play-symmetry --n=250 --a=dqn_rl-random/$(basename -- $f .json) --out="dqn-random/$(basename -- $f).json" --penalize-illegal
done


python main.py vs --game-name=nim --initial=default_initial.lp --n-initial=10 --play-symmetry --n=250 --a=dqn_rl-random --out="dqn-random.json" --penalize-illegal


echo "$B ------ VS Alpha zero $f... $NC"
FILES="./approaches/alpha_zero/saved_models/nim/not-penalized/*.json"
for f in $FILES
do
    echo "$B VS against random for $f... $NC"
    python main.py vs --game-name=nim --initial=default_initial.lp --n-initial=10 --play-symmetry --n=250 --a=alpha_zero-not-penalized/$(basename -- $f .json) --out="alpha-zero-not-penalized/$(basename -- $f).json" --penalize-illegal
done

python main.py vs --game-name=nim --initial=default_initial.lp --n-initial=10 --play-symmetry --n=250 --a=alpha_zero-not-penalized --out="alpha-zero-not-penalized.json" --penalize-illegal


FILES="./approaches/alpha_zero/saved_models/nim/penalized/*.json"
for f in $FILES
do
    echo "$B VS against random for $f... $NC"
    python main.py vs --game-name=nim --initial=default_initial.lp --n-initial=10 --play-symmetry --n=250 --a=alpha_zero-penalized/$(basename -- $f .json) --out="alpha-zero-penalized/$(basename -- $f).json" --penalize-illegal
done

python main.py vs --game-name=nim --initial=default_initial.lp --n-initial=10 --play-symmetry --n=250 --a=alpha_zero-penalized --out="alpha-zero-penalized.json" --penalize-illegal


echo ""
echo ""
echo ""
echo "$B    ---------------------$NC"
echo "$B    | Plotting results $NC"
echo "$B    ---------------------$NC"

python main.py plot --game-name=nim --file="dqn-random" --plot-out="dqn-random"

python main.py plot --game-name=nim --file="alpha-zero-not-penalized" --plot-out="alpha-zero-not-penalized"
python main.py plot --game-name=nim --file="alpha-zero-penalized" --plot-out="alpha-zero-penalized"


python main.py plot --game-name=nim --file="dqn-random.json" --file="alpha-zero-penalized.json" --file="alpha-zero-not-penalized.json" --plot-out="all"
