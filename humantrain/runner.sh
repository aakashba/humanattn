for X in KGT001 KGT002 KGT003 KGT004 KGT005 KGT007 KGT008 KGT009 KGT010
do
	for V in 0 1 2 3
	do
		echo $V
		time python3 train.py --model-type=astgnn --gpu=0 --outdir=outdir --epochs=70 --with-calcviz --batch-size=70 --with-no-savemodel --test-method=$V --test-subject=$X
	done
done
