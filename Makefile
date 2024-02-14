test:
	mv document_preprocessor.py tests
	mv indexing.py tests
	mv l2r.py tests
	mv network_features.py tests
	mv ranker.py tests
	mv relevance.py tests
	mv vector_ranker.py tests

deploy:
	mv tests/document_preprocessor.py document_preprocessor.py
	mv tests/indexing.py indexing.py
	mv tests/l2r.py l2r.py
	mv tests/network_features.py network_features.py
	mv tests/ranker.py ranker.py
	mv tests/relevance.py relevance.py
	mv tests/vector_ranker.py vector_ranker.py
