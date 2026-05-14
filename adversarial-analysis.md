# Adversarial Evaluation Analysis

## Per-hypothesis accuracy

| Hypothesis category | Correct | Total | Accuracy |
|---|---|---|---|
| negation        | 2 | 5 | 40% |
| lexical_trigger | 2 | 5 | 40% |
| domain_shift    | 3 | 5 | 60% |
| length_extreme  | 5 | 6 | 83% |
| sarcasm         | 0 | 5 | 0%  |
| other           | 3 | 4 | 75% |

## Confirmed hypotheses

**Negation** failed on 3 of 5 examples, confirming that the model does not
reliably propagate negation across a clause.

Row 4 ("The latest update did not fix any of the crashes I was experiencing.")
was predicted *neutral* at 0.44. The resolution-cue word "fix" is strongly
associated with positive outcomes in app-review training data ("they finally
fixed it"), so "did not fix" pulls the prediction away from negative toward
neutral rather than reinforcing it. The negation is not ignored outright — the
model does not predict positive — but it is not fully processed either.

Row 7 ("The whitelist option does not prevent my tree from dying when I get a
call.") was predicted *neutral* at 0.80. The verb "prevent" carries no
inherent negative valence in isolation, so negating it produces no detectable
negative signal. The model requires an overtly negative anchor word — "crash",
"broken", "hate" — not just the logical negation of a neutral verb.

Row 8 ("Never thought an app would make me less productive but here we are.")
was predicted *positive* at 0.62, the most striking failure in this category.
The informal concessive register of "but here we are" appears in positive
reviews as a self-deprecating aside, and the absence of any explicit negative
word leaves the model with no negative anchor to counteract the positive-leaning
phrasing.

**Lexical trigger** failed on 3 of 5 examples, confirming that strong
sentiment words at or near the start of a sentence dominate the prediction
regardless of what follows.

Row 11 ("I love how this app drains my battery and crashes twice a day.") was
predicted *positive* at 0.78. "Love" at the beginning of the sentence
outweighs "drains" and "crashes" later in the clause — the model's decision
is effectively made at the first strong cue word.

Row 12 ("Outstanding customer support they replied telling me the bug is by
design.") was predicted *positive* at 0.93, the highest-confidence wrong
prediction in the entire set. "Outstanding" is so strongly associated with
positive reviews in the training corpus that even an explicitly dismissive
follow-up cannot shift the prediction. This is the clearest demonstration that
the model has no representation of ironic or backhanded praise.

**Sarcasm** failed on all 5 examples, fully confirming the hypothesis that the
model cannot detect sarcasm.

Row 25 ("Oh brilliant the update removed the only feature I actually used.")
was predicted *positive* at 0.81. The exclamatory opener "Oh brilliant" mimics
the register of enthusiastic positive reviews and the model has no mechanism
to detect the ironic shift that follows.

Row 28 ("I love having to re-enter all my preferences every time there is an
update very convenient.") was predicted *positive* at 0.93. Two positive cue
words — "love" and "convenient" — bookend the sentence. The negative content
in the middle ("re-enter all my preferences every time") contains no
conventionally negative word, so the model assigns maximum positive confidence.

Row 27 ("Five stars for charging me twice for the same booking truly
exceptional service.") was predicted *positive* at 0.75. The phrase "five
stars" and the word "exceptional" are among the strongest possible positive
signals in an app-review corpus; the complaint embedded between them is
invisible to the classifier.

## Refuted hypotheses

**Length extreme** was largely refuted — 5 of 6 correct, including all three
ultra-short reviews. Row 19 ("Crashes constantly.") was correctly classified
*negative* at 0.91, and row 20 ("Works perfectly fine.") correctly *positive*
at 0.89. A single strong lexical anchor is sufficient for DistilBERT even at
three tokens, which means minimum-length inputs are not an inherent reliability
problem for this model.

The one failure in this category (row 22, the ~90-word negative review) is
the opposite of the original hypothesis: the problem is not that the model
lacks enough signal but that it has too much competing signal. The sentence
opens with positive framing ("genuinely appreciate the core idea") and
sustained negative detail follows. The model predicted *neutral* at 0.63,
averaging across the two halves rather than identifying the dominant polarity.
Length causes averaging, not noise.

**Domain shift** was partially refuted. Three of five out-of-domain sentences
were correctly classified neutral, including the recipe (row 16, 0.52) and
the sports result (row 15, 0.67). Generic factual prose with no sentiment
vocabulary causes no systematic failure — the model defaults to neutral when
no strong cue is present, which is the correct behavior.

The two domain-shift failures (rows 14 and 17) are better explained as
lexical-trigger failures than as domain-shift failures. Row 14 ("The central
bank raised interest rates by 25 basis points this morning.") was predicted
*negative* at 0.60 — "raised" followed by a numeric value resembles complaint
phrasings common in app reviews. Row 17 ("The defendant was found not guilty
on all counts after a three-week trial.") was predicted *negative* at 0.78 —
the token "guilty" is strongly associated with negative sentiment regardless
of the surrounding "not". Domain shift and lexical-trigger sensitivity
interact: out-of-domain sentences are safe unless they happen to contain a
high-weight cue token from the training vocabulary.

## What the results reveal about the decision boundary

Two specific properties emerge from these results.

The model is **token-anchored at its strongest in-vocabulary sentiment
signal**, weighted heavily toward whichever cue word carries the highest
unigram frequency in the training data, with strong recency bias toward words
near the beginning of the sentence. This explains the 0% sarcasm accuracy and
the lexical-trigger failures: "love", "outstanding", "five stars", and
"brilliant" activate the positive class at high confidence regardless of the
surrounding clause. The classifier has learned a prior from training data where
those words reliably co-occur with positive labels, and that prior is not
overridden by subsequent negative content or logical structure.

The model also **resolves ambiguity by collapsing to neutral rather than
negative**. In the majority of incorrect predictions the model predicted
*neutral* when the correct label was *negative* — rows 4, 6, 7, 9, 22, 29,
and 30. This occurred consistently when negative content was expressed through
logical structure (negation, qualification, contrast) rather than through a
negatively-valenced anchor word. The model treats the absence of a clear
positive or negative surface token as evidence for neutral, rather than
attempting to resolve the logical polarity of the sentence. In production,
this means the classifier will systematically under-detect negative reviews
written with hedged or indirect language — a meaningful real-world risk, since
dissatisfied users often phrase complaints as disappointed expectations
("I used to love this", "not as good as it was") rather than direct insults.