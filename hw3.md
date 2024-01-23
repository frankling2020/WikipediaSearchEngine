# SI 650 Homework 3
---
> Name: Haoyang Ling \
> Email: hyfrankl@umich.edu \
> UMID: 10248218


### Probem 1
Here is the bar plot showing rhe mean time-per-query-generation.
![](figures/doc2query_time.png)

```
Generating queries for doc2query/msmarco-t5-base-v1: 100%|██████████| 100/100 [00:57<00:00, 1.73it/s]
Generating queries for google/flan-t5-small: 100%|██████████| 100/100 [01:15<00:00, 1.33it/s]
Generating queries for google/flan-t5-base: 100%|██████████| 100/100 [01:35<00:00, 1.05it/s]
Generating queries for google/flan-t5-large: 100%|██████████| 100/100 [02:20<00:00, 1.41s/it]
```

**Estimated time to generate all 200K documents**: 
- `doc2query/msmarco-t5-base-v1` : 2000 * 57 s $\approx$ 2000 minutes. 
- `google/flan-t5-small`: 2000 * 75 s = 2500 minutes.
- `google/flan-t5-base`: 2000 * 95 s $\approx$ 3167 minutes.
- `google/flan-t5-large`: 2000 * 140 s $\approx$ 4667 minutes.
- However, here we only use cpu here. If we can use gpu with large batch size. It will definitely speed up. 

**Quality of Generated Queries**:
- `doc2query/msmarco-t5-base-v1` and `google/flan-t5-large` provides most question-like queries, while the rest two models most generate summary-like queries.
- `doc2query/msmarco-t5-base-v1` might outperform `google/flan-t5-large`. 
  - For [Armies in the American Civil War](https://en.wikipedia.org/wiki/?curid=50502333), `google/flan-t5-large` generates "How many of the 10 bureau chiefs were over 70 years old?", while `doc2query/msmarco-t5-base-v1` generates "what were the war forces called in the civil war".
  - Considering more than one query for each documents and the target users may be a good way to further evaluate the models. 

**Choice**: Based on the time plot and query quality, I will choose `doc2query/msmarco-t5-base-v1` because the model is fine-tuned so that it takes less time to generate queries of high quality, which is shown in the experiment results.


### Problem 2
**Encoding time**:
- `sentence-transformers/msmarco-MiniLM-L12-cos-v5`: ~179280 ms
- `multi-qa-mpnet-base-dot-v1`: ~526980 ms
- `msmarco-distilbert-dot-v5`: ~264410 ms

**Plot for MAP@10 and NDCG@10 and Encoding time**:
![](figures/bi_encoder_l2r_metrics.png)
![](figures/encoder_time.png)

**Observations**:
- `multi-qa-mpnet-base-dot-v1` outperforms the other models, but it takes the most time to encode.
- All the bi-encoders outperforms the baseline BM25.
- Performance are somehow related to time because those high-quality model usually have more parameters and therefore takes more time to encode documents.
- As for bi-encoder part, we can pre-compute the vector representatinos of each document. We are less concerned about time and more concerned about its performance.

### Problem 3
- 10 most common labels
![Alt text](image.png)

#### Summary
-  All queries produce disparate rankings across attributes.
-  Arabs rank consistently low.

#### Query 1: "person"
- **Ethnicity**:
![](figures/plots/Ethnicity_person.png)
    - The disparate rankings are observed.
    - Serbs and Greeks rank higher, while Germans,  English People, and African Americans rank lower.
    - Explain: Greek and Serbian histories might be more frequently studied and have had significant impacts on historical events.

- **Gender**:
![](figures/plots/Gender_person.png)
    - The disparate rankings are observed.
    - Gender-fluid, female, and trans woman get lower rank or less attention, while cisgender man dand male organism get higher rank or more attention.
    - Explain: Males, especially cisgender males, have historically been more documented, for example, Patrilineal society.

- **Religious**:
![](figures/plots/Religious_Affiliation_person.png)
  - The disparate rankings are observed.
  - Sunni Islam and Islam ranks higher, while Anglicanism, Catholocism, and Catholic Church ranks lower.
  - Explain: Islam is one of the largest religions with a long history and obtain worldwide attention.

- **Political**:
![](figures/plots/Political_Party_person.png)
  - The disparate rankings are observed.
  - Chinese Communist Party ranks higher, while Conservative Party ranks lower.
  - Explain: China's superpower and Chinese Communist Party plays an important role in the international relationships. 

#### Query 2: "woman"
- **Ethnicity**:
![](figures/plots/Ethnicity_woman.png)
  - The disparate rankings are observed.
  - Serbs and African Americans rank higher, while Arabs and Italians rank lower.
  - Explain: Serbian and African American women get much attention. People may like dating with Serbian woman as the first Bing search result is "Serbian Women: What Makes Them Perfect Girlfriends?". Besides, societies pay much attention to influential African American women.

- **Gender**:
![](figures/plots/Gender_woman.png)
  - The disparate rankings are observed.
  - Female and non-binary gender ranks higher, while male ranks lower.
  - Explain: it is because the query is "woman" so it would retrieve more female than male.

- **Religious**:
![](figures/plots/Religious_Affiliation_woman.png)
  - The disparate rankings are observed.
  - Hinduism ranks higher, while Catholic Church ranks lower.
  - Explain: Hinduism has diverse and rich cultural practices involving women, while Catholic Church doesn't.

- **Political**:
![](figures/plots/Political_Party_woman.png)
  - The disparate rankings are observed.
  - Bharatiya janata Party and Chinese Communist Party ranks higher, while Conservation Party and Liberal Party ranks lower.
  - Explain: Women in the BJP and Chinese Communist Party may play critical political roles.

#### Query 3: "teacher"
- **Ethnicity**:
![](figures/plots/Ethnicity_teacher.png)
  - The disparate rankings are observed.
  - Serbs ranks higher, while Arabs ranks lower.
  - Explain: Serbian culture might place a significant emphasis on the role of teachers, while Arab countries has less academic freedom.

- **Gender**:
![](figures/plots/Gender_teacher.png)
  - The disparate rankings are observed.
  - Non-binary gender and female ranks higher, while cisgender man ranker lower.
  - Explain: Teaching is often stereotypically associated with females, which might lead to higher relevance and ranking for pages related to female and non-binary gender teachers because they tend to be more patient than male.

- **Religious**:
![](figures/plots/Religious_Affiliation_teacher.png)
  - The disparate rankings are observed but less noticable than other figures.
  - Anglicanism ranks higher, while Sunni Islam ranks lower.
  - Explain: Anglicanism might has a strong historical connection with education.

- **Political**:
![](figures/plots/Political_Party_teacher.png)
  - The disparate rankings are observed.
  - Conservation Party ranks higher, while Nazi Party ranks lower.
  - Explain: The Nazi Party has a negative historical connotation related to education -- propaganda.


#### Query 4: "role model"
- **Ethnicity**:
![](figures/plots/Ethnicity_role%20model.png)
  - The disparate rankings are observed.
  - Italian ranks higher, while Germans ranks lower.
  - Explain: there are many Italian artists and famous people that fit in with role model, while Germans are less likely.

- **Gender**:
![](figures/plots/Gender_role%20model.png)
  - The disparate rankings are observed.
  - Gender-fluid ranks higher, while the cisgender ranks lower.
  - Explain: Gender-fluid individuals might be receiving more media visibility as role models in recent times.

- **Religious**:
![](figures/plots/Religious_Affiliation_role%20model.png)
  - The disparate rankings are observed.
  - Judaism ranks higher, while the Hinduism ranks lower.
  - Explain: Judaism have more widely recognized historical figure, and the Jewish community might be more actively promoting their role models, while Hinduism is strongly characterized by closure.

- **Political**:
![](figures/plots/Political_Party_role%20model.png)
  - The disparate rankings are observed.
  - Decormate Party ranks higher, while Bharatiya janata Party and Indian National Congress ranks lower.
  - Explain: Political figures from the Democratic Party in the U.S. receive more international media coverage.


#### Query 5: "professional"
- **Ethnicity**:
![](figures/plots/Ethnicity_professional.png)
  - The disparate rankings are observed.
  - Italians rank higher, while Arabs ranks lower.
  - Explain: Italy has a strong reputation in certain professional fields, such as fashion, design, and food.

- **Gender**:
![](figures/plots/Gender_professional.png)
  - The disparate rankings are observed.
  - Non-binary geder and female rank higher, while male organism ranks lower.
  - Explain: There is growing emphasis on gender diversity (especially, non-binary and female professionals) in professional settings.

- **Religious**:
![](figures/plots/Religious_Affiliation_professional.png)
  - The disparate rankings are observed.
  - Judaism and Atheism ranks higher, while Hinduism and Sunni Islam ranks lower.
  - Explain: There are many scientists related to Judaism and Atheism.


- **Political**:
![](figures/plots/Political_Party_professional.png)
  - The disparate rankings are observed.
  - Nazi Party ranks higher, while Liberal Party ranks lower.
  - Explain: Professional roles within the Nazi Party, especially in scientific and military fields, might be more extensively documented.

**Quantify the fairness of an IR ranker**:
The main idea is to find a vector representation X without the attribute information A but still can work as a good document presentation $X^*$ or X\A. 
$$
L_1 \cdot  d(X_a, X_b) \leq d(X^*_a, X^*_b),
$$
where $X_a$ and $X_b$ are the original representation, while $X^*_a$ and $X^*_b$ are vector representations without attribute information. It would be better if we have
$$
d(X^*_a, X^*_b) \leq L_2 \cdot d(X_a, X_b)
$$
Therefore, we can actually see the variance of the group mean $\{\mu_k\}$ based on this attribute information can serve as a metric for fairness:

$$
X^* = \mathrm{argmin_{X} \sum_{i=1}^k \dfrac{n_k \|\mu - \mu_k\|^2}{\sigma^2}}
$$

For each term, it serve a normalized metric for how the  group mean deviates from the total mean.


### Problem 4
I choose 5 queries from the development data. They are:
1. "What is the history and cultural importance of traditional Chinese martial arts"
2. "How are countries responding to the challenges of misinformation and disinformation campaigns"
3. "Analyze the role of architecture in promoting sustainability and green design"
4. "How do colleges and universities ensure campus safety and security"
5. "How have smartphones influenced the gaming industry and mobile gaming trends"

**Spearman's correlation**: 0.5829

**Scatter plot**
![](figures/bi_cross_rank_scatter.png)
As it is difficult to extract information from scatter plot, I also draw a hex plot for convenience. 

**Hex plot**
![](figures/bi_cross_rank_hex.png)

**Descriptions**: 
From the figure, we can see that for document rank around 1 - 25 in bi-encoder, they will be most likely to be ranked around 50 - 100 in cross encoder, which is indicated by the darkness of each hex.

**Observations**:
- Similarities: It can be seen that the two models have most agreement for the set of the top 150 document as the hex in the diagonal are darker than the rest though they may disagree with each other in the inner order of these 150 documents. 
- Differences: Hoever, there are some data points in the top left and bottom right, which indicates the discrepency between the two models. Data points are not closely distributed along the diagonal.

### Problem 5
![](figures/pipeline_metrics.png)

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <td></td>
      <td>map</td>
      <td>ndcg</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>base</td>
      <td>0.043913</td>
      <td>0.332071</td>
    </tr>
    <tr>
      <td>l2r</td>
      <td>0.070226</td>
      <td>0.343751</td>
    </tr>
    <tr>
      <td>vector_ranker</td>
      <td>0.059544</td>
      <td>0.314000</td>
    </tr>
    <tr>
      <td>new_model</td>
      <td>0.059102</td>
      <td>0.333798</td>
    </tr>
  </tbody>
</table>
</div>

- Adding deep learning-based features help us improve from baseline BM25 as pure vector ranker outperforms BM25 in MAP but not in NDCG. And the cross encoder features help us improve NDCG without lowering MAP too much.
- But, it still can't beat our hw2 models. It may be the poor quality of training data as some queries don't have document with score higher than 3 (namely, all negative samples).

### Ablation Study
- BM25 (hw2)
  - None: base
  - l2rranker: l2r
- l2rranker + index_augment + cross_encoder
  - BM25: pipeline_1
  - VectorRanker: pipeline_2
- vector_ranker + l2rranker
  - index: pipeline_3
  - index_augment: pipeline_4
- vector_ranker + l2rranker + index_augment
  - None: pipeline_4
  - cross_encoder: pipeline_2

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th></th>
      <th>MAP@10</th>
      <th>NDCG@10</th>
      <th>CONFIG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>base</td>
      <td>0.043913</td>
      <td>0.332071</td>
      <td> index + BM25</td>
    </tr>
    <tr>
      <td>BM25</td>
      <td>0.039091</td>
      <td>0.331340</td>
      <td> index_aug + BM25</td>
    </tr>
    <tr>
      <td>VectorRanker</td>
      <td>0.059544</td>
      <td>0.314000</td>
      <td> index_aug + vec_rank</td>
    </tr>
    <tr>
      <td>l2r</td>
      <td>0.070226</td>
      <td>0.343751</td>
      <td> index + BM25 + l2r </td>
    </tr>
    <tr>
      <td>pipeline_1</td>
      <td>0.052045</td>
      <td>0.330412</td>
      <td> index_aug + BM25 + l2r (cross_enc) </td>
    </tr>
    <tr>
      <td>pipeline_2</td>
      <td>0.059102</td>
      <td>0.333798</td>
      <td> index_aug + vec_rank + l2r (cross_enc) </td>
    </tr>
    <tr>
      <td>pipeline_3</td>
      <td>0.046063</td>
      <td>0.328585</td>
      <td> index + vec_rank + l2r </td>
    </tr>
    <tr>
      <td>pipeline_4</td>
      <td>0.056367</td>
      <td>0.331492</td>
      <td> index_aug + vec_rank + l2r </td>
    </tr>
  </tbody>
</table>
</div>

![](figures/pipeline_metrics_extra.png)

**Discussion**
- Document augmentation might help the re-rank, but a further experiment shows that pure BM25 with document augmentation gets lower MAP in the first matching part than that without augmentations.
- VectorRanker outperforms BM25, which shows that bi-encoder are providing important information.
- Cross encoders will increase NDCG score without lowering MAP too much. It shows the effectiveness to use CrossEncoder in the reranking part.
- HW2 pipeline performs better than others, which shows that the added new features in hw2 plays a quite critical role in prediction.