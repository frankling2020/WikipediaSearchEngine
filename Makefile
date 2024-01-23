test:
	mv document_preprocessor.py hw4-tests
	mv indexing.py hw4-tests
	mv l2r.py hw4-tests
	mv network_features.py hw4-tests
	mv ranker.py hw4-tests
	mv relevance.py hw4-tests
	mv vector_ranker.py hw4-tests

deploy:
	mv hw4-tests/document_preprocessor.py document_preprocessor.py
	mv hw4-tests/indexing.py indexing.py
	mv hw4-tests/l2r.py l2r.py
	mv hw4-tests/network_features.py network_features.py
	mv hw4-tests/ranker.py ranker.py
	mv hw4-tests/relevance.py relevance.py
	mv hw4-tests/vector_ranker.py vector_ranker.py
