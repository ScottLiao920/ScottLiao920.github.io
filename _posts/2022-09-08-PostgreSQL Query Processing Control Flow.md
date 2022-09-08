---
title: PostgreSQL Query Processing Control Flow
categories:

- General
  feature_image: "https://picsum.photos/2560/600?image=872"

---

This is a sketch of control flow for full query processing:

	CreateQueryDesc

	ExecutorStart
		CreateExecutorState
			creates per-query context
		switch to per-query context to run ExecInitNode
		AfterTriggerBeginQuery
		ExecInitNode --- recursively scans plan tree
			ExecInitNode
				recurse into subsidiary nodes
			CreateExprContext
				creates per-tuple context
			ExecInitExpr

	ExecutorRun
		ExecProcNode --- recursively called in per-query context
			ExecEvalExpr --- called in per-tuple context
			ResetExprContext --- to free memory

	ExecutorFinish
		ExecPostprocessPlan --- run any unfinished ModifyTable nodes
		AfterTriggerEndQuery

	ExecutorEnd
		ExecEndNode --- recursively releases resources
		FreeExecutorState
			frees per-query context and child contexts

	FreeQueryDesc


