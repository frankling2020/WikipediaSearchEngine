test:
	mv document_preprocessor.py hw5-tests
	mv indexing.py hw5-tests
	mv l2r.py hw5-tests
	mv network_features.py hw5-tests
	mv ranker.py hw5-tests
	mv relevance.py hw5-tests
	mv vector_ranker.py hw5-tests

deploy:
	mv hw5-tests/document_preprocessor.py document_preprocessor.py
	mv hw5-tests/indexing.py indexing.py
	mv hw5-tests/l2r.py l2r.py
	mv hw5-tests/network_features.py network_features.py
	mv hw5-tests/ranker.py ranker.py
	mv hw5-tests/relevance.py relevance.py
	mv hw5-tests/vector_ranker.py vector_ranker.py
