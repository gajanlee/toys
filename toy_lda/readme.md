# LDA

### Usage

* use lda datasets
```python
from lda import datasets

vocab = datasets.load_reuters_vocab()
X = datasets.load_reuters()

docs = [[] for _ in range(X.shape[0])]
for i, doc in enumerate(X):
    term_ids = np.nonzero(doc)[0]
    for term_id in term_ids:
        docs[i] += [vocab[term_id]] * doc[term_id]
    
model = LDA()
model.fit_documents(docs)
model.inference(docs)
model.display_distribution()
```

### Performance

* iteration 1

```
+----------+-----------+-----------+---------+-----------+----------+---------------+---------+-----------+--------------+-----------+
| topic_id |    top1   |    top2   |   top3  |    top4   |   top5   |      top6     |   top7  |    top8   |     top9     |   top10   |
+----------+-----------+-----------+---------+-----------+----------+---------------+---------+-----------+--------------+-----------+
|  topic0  |   church  | president |   last  |    1981   | catholic |      city     |  mother | officials |     pope     |   world   |
|  topic1  |   church  |    king   |   time  |    city   | clinton  |     years     |   very  |   people  |    since     |  election |
|  topic2  |   years   |    u.s    | charles |   people  |  church  |      take     |   year  |   state   |     made     |    time   |
|  topic3  |    pope   |   people  |  mother |   church  |  teresa  | international |  prime  |   party   |     city     |   paris   |
|  topic4  |   people  |    film   |  church |    pope   |  elvis   |      week     |   life  |   party   |    years     |    last   |
|  topic5  |   church  |   first   |   last  |   years   |   pope   |    catholic   |  mother | reporters |    death     |   russia  |
|  topic6  |    pope   |   church  |   u.s   |  charles  | million  |     first     |   year  |    life   |    world     |  catholic |
|  topic7  |   first   |   church  |   told  |    pope   |  three   |     years     | charles |     us    |    royal     |   mother  |
|  topic8  |    pope   |   world   |   n't   |   church  |   told   |      own      | carried |  germany  |   against    |  minister |
|  topic9  |   church  |   years   |  teresa |    set    |   year   |     during    |  first  |    time   |    public    |    last   |
| topic10  |   church  |    pope   |  years  |    war    | american |    hospital   |  france |  catholic |   against    |    last   |
| topic11  |   church  |   mother  |   pope  | statement |  united  |    minister   |  become |   first   |  territory   |    told   |
| topic12  |    pope   |  country  |   few   |   church  |  sister  |      film     |  family |    year   |    leader    | operation |
| topic13  | president |    pope   |  church |  british  |  years   |    harriman   |  saying |    year   | conservative |   house   |
| topic14  |  british  |    year   |  church |  ceremony |   last   |      pope     |  years  |   world   |   tuesday    |   bowles  |
| topic15  |   church  |   france  |  german | political |   age    |     years     | charles |   royal   |  president   |  harriman |
| topic16  |   church  |    died   |  world  |    told   |  people  |     mother    |  since  | president |    prince    |   added   |
| topic17  |    told   |  yeltsin  |  years  |  vatican  |   day    |    harriman   |  prize  |    life   |   michael    |    pope   |
| topic18  |   church  |    told   |  former |   roman   |   pope   |     mother    |   king  |    last   |     made     |   world   |
| topic19  |   church  |    pope   |  first  |    last   |   both   |     month     |   year  |    time   |     john     |  catholic |
+----------+-----------+-----------+---------+-----------+----------+---------------+---------+-----------+--------------+-----------
```

* iteration 5
```
+----------+----------+------------+-----------+------------+-----------+-----------+-----------+--------------+----------+----------+
| topic_id |   top1   |    top2    |    top3   |    top4    |    top5   |    top6   |    top7   |     top8     |   top9   |  top10   |
+----------+----------+------------+-----------+------------+-----------+-----------+-----------+--------------+----------+----------+
|  topic0  |   pope   | christian  |   timor   |    belo    |   during  |   denied  |   month   |   funeral    |  award   | gemelli  |
|  topic1  |  church  |   mother   |    poor   |   teresa   |   recent  |  calcutta |    john   |   election   |   very   | birthday |
|  topic2  |  diana   |   public   |   parker  |   years    |  vatican  |  charles  |  american |    people    | marriage |  bowles  |
|  topic3  |   city   | operation  |   death   |   world    |   later   |    war    |    fans   |     own      |   want   |   life   |
|  topic4  |   pope   |   church   |  romania  |  million   |   moscow  |    part   | operation |    never     | catholic | thursday |
|  topic5  |  church  |   years    |    u.s    |  simpson   |   month   |   death   |    born   |     take     | children |  state   |
|  topic6  |  police  |  cunanan   |   miami   |    year    |   church  |  versace  |   south   |     four     | million  |  beach   |
|  topic7  |  royal   | political  |  germany  |  against   |  several  |    left   |  angeles  |    public    |  years   |   jews   |
|  topic8  |   pope   |   world    |    time   |   united   |   first   |  minister |    very   |     say      |  roman   |   real   |
|  topic9  |   pope   |   peace    |    east   |  doctors   |    last   |   prize   |  surgery  |   hospital   |   year   |  order   |
| topic10  |  elvis   |   church   |  against  | exhibition |   human   |   years   |  concert  |     take     |  music   | america  |
| topic11  |  order   |     go     |    u.s    |    city    |   india   |    last   |  germany  |    famous    |   year   | chicago  |
| topic12  | yeltsin  |   family   |   party   |    king    | president |  country  |    last   |     bill     |  people  | russian  |
| topic13  |  teresa  |    home    |   mother  |   house    |   people  |    nuns   |    york   |    sister    | doctors  |   art    |
| topic14  |  mother  |    made    |  catholic |   sunday   |    last   |    life   |   woman   |     rome     |  local   |   days   |
| topic15  | harriman |  clinton   | president |   church   |   bishop  | bernardin |   paris   |   cardinal   |  wright  | british  |
| topic16  |  while   | ambassador |    died   | churchill  |   former  |  tuesday  |    went   |   letters    |  queen   |  church  |
| topic17  |  mother  |   prince   |    king   |  charity   |   heart   |   among   |   visit   | missionaries |  prime   |  years   |
| topic18  | hospital |  tuesday   |    told   | condition  |    life   | wednesday |    time   |    teresa    |  first   | husband  |
| topic19  | charles  |  camilla   |   prince  |   queen    |    died   |   bowles  |  minister |    quebec    |  people  |  irish   |
+----------+----------+------------+-----------+------------+-----------+-----------+-----------+--------------+----------+----------+
```