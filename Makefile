test:
	mv src/document_preprocessor.py tests
	mv src/indexing.py tests
	mv src/l2r.py tests
	mv src/network_features.py tests
	mv src/ranker.py tests
	mv src/relevance.py tests
	mv src/vector_ranker.py tests

deploy:
	mv tests/document_preprocessor.py src/document_preprocessor.py
	mv tests/indexing.py src/indexing.py
	mv tests/l2r.py src/l2r.py
	mv tests/network_features.py src/network_features.py
	mv tests/ranker.py src/ranker.py
	mv tests/relevance.py src/relevance.py
	mv tests/vector_ranker.py src/vector_ranker.py
