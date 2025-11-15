localbench:
	@python test_triton_local.py 2>&1 |tee a.log
	
cudabench:
	@CUDA_VISIBLE_DEVICES=1 python test_triton_local.py 2>&1 |tee a.log

cuda_localcompute:
	@rm -rf ~/.triton/cache && CUDA_VISIBLE_DEVICES=1 compute-sanitizer --tool memcheck --print-level info python test_triton_local.py 2>&1 |tee a.log


amd_local8:
	@bash kill.sh && python run.py 2>&1 |tee a.log


print:
	@touch a.txt
       


gen: 
	@git add -u && git commit -m "$(ARGS)" && git push origin --all && git push origin --tags && python gen.py


