<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>
    
<script type="text/javascript"
        src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

# 1. Review: Statistics, Correlation, Regression, Gradient Descent

## 1.1 Formulating experiments and terminology

**Hypothesis testing**
The essence of hypothesis testing can be boiled down to the scenario where we have a claim and corresponding data that could help us evaluate its accuracy. In short, we want to answer the questions whether the data supports the claim or not. The claim is called the “hypothesis," and so the procedure is called “hypothesis testing."

The **treatment variable**, also called the *independent variable*, is what we are able to modify and is what causes the changes to other variables.

The **outcome variable**, also called the *dependent variable*, is what we observe and is what is affected by the treatment variable.

**Patient selection**: In general, how we select the treatment and control groups will influence the population for which the conclusion is valid.

**Control group**: We need to compare the outcome variable for those who have received the treatment with a baseline (i.e, the control group). Here, the control group (who were not offered a treatment) must be a comparable set of people to the treatment group (who have been offered a treatment).

**Features of the patients**: One way to make accurate comparison and interpretation of results is to ensure that the treatment group is representative across factors such as health status, age, and ethnicity, and is similar along these dimensions as the control group. This way, we can attribute any differences in outcome to the treatment rather than differences in covariates. In a general study, it is upon the researchers' discretion to determine which features to be careful about.

An approach that addresses the three points above is called the **Randomized Controlled Trial (RCT)**. A Randomized Controlled Trial (RCT) is an experimental design in which treatments are assigned at random.

The treatment is applied to the treatment group, while the control group are either simply set aside for observation or are given a placebo treatment. <mark>Then, by the Law of Large Numbers, we will expect the difference in averages of any relevant feature between the two groups to be fairly small. This allows us take the difference of the mean of the outcome variables in the two groups in order to estimate the treatment effect</mark>.

**Stratification**: One common modification to a RCT is to use stratification. Here, we pre-divide the population into groups and then sample proportionately from each group. This allows us to not leave the unbiasedness of our sampling, with regards to a particular classification or feature, up to chance. Stratification also enables subgroup analysis, which is analyzing the treatment effect within a particular group.

**Sample size** concerns: In subgroup analyses, and in RCT's where our sample is small, however, we wil likely run into sample size issues, which could affect the precision of the estimate. When we are only sampling a small number of points, it is more likely for us to either hit a large number of consecutively large or consecutively small points, as compared to when our sample is large. Thus, a particular difference in outcome means will in some sense mean less in a small sample. <mark>We need a relatively large sample to detect a treatment effect that is small</mark>.

**Double blindness**: any experiment that involves human subjects, factors related to human behavior may influence the outcome, obscuring treatment effects. For example, if patients in a drug trial are made aware that they actually received the new treatment pill, their behavior may change in a number of ways, such as by being more or less careful with their health-related choices. Such changes are very difficult to model, so we seek to minimize their effect as much as possible. 

The standard way to resolve this is through a double-blind study , also called a blinded experiment . Here, human subjects are prevented from knowing whether they are in the treatment or control groups. At the same time, whoever is in charge of the experiment and anyone else who could interact with the patient are also prevented from directly knowing whether a patient is in the treatment or the control group. This is to prevent a variety of cognitive biases such as [observer bias](https://en.wikipedia.org/wiki/Observer_bias) or [confirmation bias](https://en.wikipedia.org/wiki/Confirmation_bias) that could influence the experiment.

<mark>In situations where randomization is impossible, our best option is an observational study</mark>, where we attempt to estimate a causal effect from existing data. For example, when studying the effects of particular genes in humans, editing genes is considered unethical and hence not done. There are methods such as instrumental variables that allows us to use the causal effect from one treatment to estimate another relationship that we are more interested in. These methods, however, are out of scope for this course. (The course [14.310x Data Analysis for Social Scientists](https://mitxonline.mit.edu/courses/course-v1:MITxT+14.310x/) gives a fairly detailed overview of this technique.)

**Confounding variables**: As the treatment is not randomized, we are unable to assess the causal effect by taking a difference in means. A bias in the difference in means arises when, even without any treatment, the group that are more likely to be treated have a different baseline outcome than the group that are less likely to be treated. As a result, we are unable to discern whether the difference is from the treatment's effect or from a difference in baseline values. This phenomenon is called confounding, and <mark>any external variable that can be identified to influence both the treatment and the outcome variables is called a confounding variable</mark>. To resolve this problem we can use:

  - **Stratification**: The stratification approach discussed earlier is a possible remedy to this. If the confounding variable is a demographic category, then balancing the treatment variable within each category resolves the issue. In the smoking example, a confounding variable is that people who are older both smoke more and are more susceptible to lung cancer. Then, sampling an equal number of smokers and non-smokers within each age group corrects for this confounding variable.
  - **Control variables**: The use of control variables is another fix to this issue. The study of control variables in the context of causality is complex, so we give a brief overview of what makes an appropriate control variable. In an ideal setting, we select controls that will capture all possible sources of bias, factors that lead to both the treatment and outcome values. Most studies will only identify a subset of these sources, however. After identifying the controls, there are a whole array of techniques to account for their effects, the most common of which is multivariate regression. For example, consider a study that investigates the effect of years of education on income at age 30. We expect someone from a well-off family to stay in school longer for many reasons, one of which is the high cost of higher education. At the same time, one's socioeconomic status while growing up affects future income for reasons unrelated to education, such as through higher expectations and better informal networks. Thus, family income is an appropriate control.

**Control variable pitfalls and causality**: Consider a study that investigates the effect of smoking on stroke incidence, as discussed in the video. Then, suppose hospitalization is proposed as a control variable, meaning that, for example, we conduct the observational study with patients in a particular hospital as our sample. This time, the control may in fact introduce bias, because both smoking and stroke can cause hospitalization, the former, albeit more indirectly. This will negatively bias the estimated relationship, because if we condition on hospitalized patients, the effect of one cause will crowd out the effect of the other.

To further illustrate why control on a common descendant variable in a casual diagram will lead to inference bias, we can see another simpler example: Let $X$ and $Y$ be independent Bernoulli random variables (e.g. coin flips) and let $Z = (X \space XOR \space Y)$. The relationship between $X$, $Y$ and $Z$ can be summarized in this causal graph: $X$ > $Z$ < $Y$ where we have no arrow from $X$ to $Y$ or vice versa because $X$ and $Y$ are independent.

Now suppose we want to find out if $X$ and $Y$ are independent. If we look at many realizations of $X$ and $Y$ we will come to the correct conclusion that the two are indeed independent. However, if we condition on $Z$, we will conclude that $X$ and $Y$ are negatively correlated so that the estimated effect of $X$ on $Y$ is negatively biased. To see why this is the case notice that if $Z = 1$, then whenever $X = 0$, $Y$ must be $1$. $Z$ here is a bad control because conditioning on it opens a so called "back-door" path from $X$ to $Y$.

**Example**

Suppose the data is generated following the causal graph below. In a causal graph, if there is an edge from $A$ to $B$, then variable $A$ has an direct effect on variable $B$.

<center><img src="Images\images_causal_3.png" alt="Alt Text" width="400" height="300"></center>

A Causal Graph. In this graph, each node is a random variable, and an edge from $A$ to $B$ means the variable $A$ has direct effect on $B$.
Recall that a confounder is a variable that influences both the dependent variable and independent variable. We are interested in the effect of $X$ on $Y$. Which of the variable in the following causal graph is a confounder for variable $X$ and $Y$? (Choose all that apply.). 

- **Resp: $A$ and $E$**.

Suppose we want to estimate the direct relation between  and  from data. What should we do? (Choose any acceptable answer.)

- **Resp: Control for confounders $A$ and $E$ and intervening variable $D$, and estimate relation between $X$ and $Y$.**

## 1.2 Example case

**Experiment Setup**\
Let's first recap the experimental design for the HIP study. The population is the set of 700,000 female enrollees in the insurance program. For this experiment, 62,000 women were selected randomly and split into two groups: 31,000 in the control and 31,000 in the treatment groups. Those in the control group receive the standard healthcare as part of the insurance plan, while those in the treatment group in addition are offered four free mammographies (breast cancer screens) a year for five years; if anything abnormal were detected, she got early treatment. She could also choose to reject the screening.

**Examining Results**\
In this experiment, 10,800 women rejected the screening while rest of 20,200 women accepted the screening. The main quantity of interest for each group is the death rate from breast cancer in the five years of follow-up. The number in each group and the rate are shown in the third column of the table, under the label "Breast cancer." Aside from this rate, the study also took note of the death rate from all other causes aside from breast cancer. This is similarly shown in the fourth column of the table. Later on, we will discuss why this statistic is also important.

<center><img src="Images\HIP_Test.png" alt="Alt Text" width="550" height="300"></center>

How do we use the results from the table? If we want to show that being in the treatment group has an effect on reducing deaths due to breast cancer, then we clearly have to compare the death rate between this group and the control. To counter-check, however, we have to argue that the two groups are similar enough in terms of risk factors related to mortality. Indeed, <mark>if the rate of death from all other causes, as seen in the fourth column, is much higher in one group than another, then this is evidence that there may be a substantial difference</mark>. This difference may lead to different base rates of breast cancer deaths, which in turn make a direct comparison of the death rate inaccurate.

**Using other metrics in the table to assess a comparison**

The fourth column of the chart shows us why the only appropriate comparison is the full (Total) Treatment group with the Control group. The death rates due to all other cases are 27 and 28 per 1000, which are not very far. On the other hand, a common error is to compare the Screened and the Control. For these two groups, the All Other death rates are 21 and 28 per 1000 respectively, which are significantly further apart. (For now, we are using a subjective judgement to decide on whether the All Other death rates are significantly different. The methods in the following videos will apply to this as well.)

What causes the large difference in the rates of All Other causes of death between the Screened group and the Control group? <mark>It is likely that conditional on being offered the screenings, whether a patient accepted the screenings or not gives information about their mortality rates</mark>. For example, those who accepted the mammography may tend to care more about their health and hence are less likely to die of other natural causes. It could also be that those already undergoing treatment for another disease will refuse due to possible complications. <mark>In short, we want there to be no distinguishing factor between the groups we compare </mark>.

**Causal Effect Identification and Calculation**

Finally, we calculate the causal effect and also more precisely define what it represents. This is an **intention-to-treat analysis**, which focuses on the intention of offering a treatment rather than the actual treatment itself. In such an analysis, we compare the whole treatment group (everyone offered) against the whole control group. From the table, we can get the following figures:

- Death rate from breast cancer in control group: $0.00203$ ($= \frac{63}{31000}$)
- Death rate from breast cancer in treatment group: $0.00126$ ($= \frac{39}{31000}$)

Hence, we can estimate the treatment effect to of offering mammography to be the difference in death rates: $0.00203 - 0.00126 = 0.00077$, or around $0.77$ deaths per 1000.

## 1.3 Distribution models

As an interlude, we will review the Bernoulli and binomial models. The Bernoulli model describes discrete events or their corresponding indicator variables, while the binomial model describes the sum of a fixed number of independent indicator variables. Both models are based on a probability parameter , which corresponds to the probability that a given event occurs.

An **indicator variable** is a random variable that has a corresponding event. The indicator variable takes on the value $1$ if the event occurs and the value $0$ if the event does not occur.

## 1.3.1 Bernoulli distribution

Bernoulli random variables are used to model random experiments with only two possible outcomes. In our example, an individual in the mammography study can experience only two possible outcomes: death attributed to breast cancer, represented by the outcome $1$, or not, represented by $0$.

A Bernoulli random variable with parameter $p$ is a random variable that takes the value $1$ with probability $p$ and the value $0$ with probability $1-p$. The Bernoulli distribution is the discrete probability distribution of a Bernoulli random variable. Hence, we can write the Bernoulli probability mass function (pmf) as:

$$
f(x) = p, \space\space if \space\space x=1 \\
f(x) = 1-p, \space\space if \space\space x=0
$$

$$
f(x) = p^{x}(1-p)^{1-x} = px + (1-p)(1-x)
$$

where the bottom expression are alternate forms that could make calculations more tractable. From this pmf, we can derive its expectation to be $p$, the parameter itself, and its variance to be $p - p^{2} = p(1-p)$.

In many applications of the Bernoulli model with multiple indicator variables, <mark>they are independent and identically distributed (i.i.d.)</mark>, meaning that the indicator variables are mutually independent and that they are all Bernoulli with the same parameter $p$. The Binomial distribution, which will be discussed later, models this special case of multiple Bernoulli random variables.

**Exercise**:

Consider an application of the Bernoulli model to the mammography study, and index the patients in the control group from 1 to 31000. Define $X_1,...,X_31000$ to be random variables where $X_i$ is an indicator variable for whether patient $i$ died of breast cancer. Suppose that our model for breast cancer deaths is as follows: $X_1,...,X_31000$ ~ (i.i.d.) Bernoulli($p$). Which of the following statements correctly describes the model?

- **Resp: In the model's data generating process, each participant has a probability  of dying from breast cancer and each participant's death is independent from other participants.**

## 1.3.2 Binomial distribution

<mark>The binomial random variable with parameters $n$ (trials) and $p$ (probability) is defined as the sum of $n$ independent Bernoulli random variables, all with parameter $p$ </mark>. In the mammography example, if we model the deaths due to breast cancer in the control group by $X_1,...,X_31000$ ~ (i.i.d.) Bernoulli($p$), <mark> then the number of such patient deaths can be modelled as a binomial random variable with parameter $p$. In statistical language, we say that a binomial random variable is the sum of independent Bernoulli trials, where each Bernoulli trial represents a single indicator variable</mark>. From the definition, we can compute the probability mass function (pmf) of a binomial random variable $Y$ with parameters $n$ and $p$ to be:

$$
f(Y = k) = \binom{n}{k}p^{k}(1-p)^{n-k}
$$

where
- $\binom{n}{k}$ is the number of different ways of choosing $k$ out of the $n$ Bernoulli variables.
- $p^{k}$ is the probability that a given set of $k$ Bernoulli random variables all take value 1.
- $(1-p)^{n-k}$ is the probability that a given set of $n-k$ Bernoulli random variables all take value 0.

## 1.3.3 Modelling breast cancer deaths in the control group

We now model the number of breast cancer deaths in the control group as a binomial variable. First, since there are 63 deaths out of 31000 patients in the control group, we can estimate the parameter ($p = \frac{63}{31000} = 0.00203$) (or 2.03 per 1000). That is, each patient's death is modeled as a Bernoulli variable with parameter $p$. The total number of breast cancer deaths is then a binomial model with 31000 Bernoulli trials and probability 0.00203 for each trial.

<center><img src="Images\BinomialDistribution.png" alt="Alt Text" width="450" height="300"></center>

**Exercise**

Consider a binomial model where each of 31000 individuals has a probability of 0.00203 (or 2.03 per 1000) of dying due to breast cancer. Use the binomial pmf to calculate the probability that exactly 63 of the 31000 patients die of breast cancer.

```python
# Importing libraries
import numpy as np
from scipy.special import comb

# Defining number of samples / runs
n = 31000

# Defining number of positive cases (1)
k = 63

# Defining probability of event happening
p = 0.00203

# Calculating binomial pmf
pmf = comb(n, k)*(p**k)*((1-p)**(n-k))

# Printing pmf
pmf

0.05024468664277072
```

## 1.3.4 Poisson distribution

The Poisson random variable is based on taking the limit of a binomial distribution with a fixed mean $np$. As we take $n \rarr \infty$, the distribution converges to a fixed discrete pmf which we parameterize by $\lambda = np$. Indeed, we can compute the probability of $k$ successes, substitute $p = \lambda/n$, and then take the limit.

$$
\lim_{n \rarr \infty} \binom{n}{k}p^{k}(1-p)^{n-k} = \lim_{n \rarr \infty} \frac{n!}{k!(n-k)!}p^{k}(1-p)^{n-k}
$$

$$
= \lim_{n \rarr \infty} \frac{n(n-1)...(n-k+1)}{k!}\left(\frac{\lambda}{n}\right)\left(1-\frac{\lambda}{n}\right)^{n-k}
$$

$$
= \lim_{n \rarr \infty} \left(\frac{n(n-1)...(n-k+1)}{n^{k}}\right)\frac{\lambda^{k}}{k!}\left(1-\frac{\lambda}{n}\right)^{n-k}
$$

$$
= (1)\frac{\lambda^{k}}{k!}e^{-\lambda}
$$

$$
= \frac{e^{-\lambda}\lambda^{k}}{k!}
$$

<mark>herefore, when data follows binomial distribution with large $n$ (number of trials) and small $p$ (probability of success), Poisson($np$) is a good approximation to Binomial(n, p).</mark>

Another interpretation of the Poisson random variable is in terms of a random process called the **Poisson process**. This is defined as a process where events can occur at any time in continuous time, with an average rate given by the parameter $\lambda$ and satisfying the following conditions:

- Events occur independently of each other.
- The probability that an event occurs in a given length of time is constant.

**Example**

**Application of the Poisson distribution**: We again consider the mammography study. In the control group, there are 31000 patients and we estimated the probability of each patient dying from breast cancer to be 0.00203.

Let $n=$ 31000 be number of independent Bernoulli trials and $p = $ 0.00203 be a constant probability of breast cancer death. We can then approximate the number of deaths as Poisson, under the following assumptions:

- One individual's death from breast cancer is independent from every other individual's death from breast cancer.
- The probability that a death occur in a five year window does not change over time.
- Binomial(31000, $p$) is well-approximated by Poisson(31000$p$), which is justified by Binomial(31000, 0.00203) has a large $n$ and a small $p$.

From the approximation definition, the plot of the pmf of Poisson(63) is thus very similar to the plot of the pdf of Binomial(31000, $p$).

<center><img src="Images\PoissonDistribution.png" alt="Alt Text" width="400" height="300"></center>

## 1.4 Hypothesis testing

## 1.4.1 Motivation for hypothesis testing

Going back to the HIP mammography study, recall that we are interested to find out whether offering a mammography for early detection reduces deaths due to breast cancer. This is an intent-to-treat analysis, where the treatment variable is whether a patient is offered mammography, as this is what is relevant for policy purposes. From the study with 31000 patients in each of the control and treatment groups, data is collected and summarized in the table below.

<center><img src="Images\HIP_Test.png" alt="Alt Text" width="550" height="300"></center>

We can see that the death rate per 1000 women due to breast cancer goes down from 2.0 in the control group to 1.3 in the treatment group. <mark>While this is a sizeable reduction, due to variance in different datasets, it may be the case that this reduction happened just by chance</mark>. Indeed, it is possible that if we are to repeat the study, the treatment group may now have a higher death rate. The role of hypothesis testing is to assess how significant the change was in the death rate.

## 1.4.2 Hypothesis testing and modelling overview

A high-level summary of hypothesis testing is that it involves calculating the probability, under a given model, that an observation equal to or more extreme than what is observed in the treatment group is obtained, conditioned on the treatment having no effect. In the mammography study, we wish to calculate the probability that the treatment group has an observed death rate of 0.0013 or below, assuming that each patient in the treatment group has the same probability of death as in the control group.

The role of a statistical model is in calculating this probability. Without a model and its corresponding assumptions, we cannot determine how likely a particular observation in the treatment group is. In the mammography study, we use a Bernoulli model for the individual deaths, with the additional assumption that the deaths are independent and identically distributed. A Bernoulli model has a parameter, which we call $pi$ in this application. We can thus write the model, on the individual level, as:

$$
X_1, X_2, ..., X_31000 \space\space (i.i.d.) \space\space \sim \space\space Bernoulli(\pi)
$$

In this example, we use the Poisson approximation of the sum of independent Bernoulli random variables, and the parameter $\lambda$ for the Poisson approximation is based on the sum of the expected values of the Bernoulli r.v.'s. Since there are 31000 of them, each with parameter $\pi$, the parameter is then $\lambda = 31000 \pi$. Hence, we can model the total number of deaths due to breast cancer in the treatment group as:

$$
Y = X_1 + X_2 + ... + X_31000 \space\space \sim \space\space Poisson(31000\pi)
$$

Now, we formalize and generalize the hypothesis testing setup. We define two contrasting statements that together summarize the hypothesis testing objective. They are the **null hypothesis** and the **alternative hypothesis**. In a treatment-control experimental setting, they take on the following roles:

- Null hypothesis $H_0$: claim that the treatment **does not** have a significant effect on the outcome, also known as the status-quo.
- Alternative hypothesis $H_A$: claim that the treatment **does** have a significant effect on the outcome.

When we have a parameteric statistical model, $H_0$ and $H_A$ can be formulated in terms of restrictions on the parameter. Indeed, in the mammography study, we can formulate the two contrasting claims on whether mammography is effective. This formulation is based on comparing the parameter $\pi$ in the treatment group to the estimated parameter of 0.00203 in the control group.

- Null hypothesis $H_0$: $\pi = $ 0.00203 (or equivalently, $\lambda = 63$), implying that offering mammography did not affect the breast cancer death rate.
- Alternative hypothesis $H_A$: $\pi < $ 0.00203 (or equivalently, $\lambda < 63$), implying that offering mammography had the effect of decreasing the breast cancer death rate.

In general, if we have a parametric statistical model with parameter $\theta$, the null and alternative hypotheses are expressed as claims that the parameter is in sets $\Theta_0$ and $\Theta_A$ respectively. These two sets are assumed to be disjoint, but their union need not be equivalent to the whole parameter space. We do not impose this last requirement because the focus of hypothesis testing is to compare which of the two hypotheses is more likely to be the case, and we may ignore parameter values that are outside our scope of interest.

## 1.4.3 Hypothesis testing and the test statistic

Hypothesis testing involves distinguishing between the **null hypothesis** and the **alternative hypothesis**. Under the null hypothesis, we have a baseline distribution (or set of distributions) of the observation from the model and the corresponding parameter(s). Based on the observation, we decide whether or not to **reject the null hypothesis**.

- We **reject the null hypothesis** if we deem it relatively unlikely for the null hypothesis to be true, given the observations.
- We **fail to reject the null hypothesis** if we do not have sufficient evidence from the observation to discredit the null hypothesis.

Notice the asymmetry between the two hypotheses. The conclusion from hypothesis testing is whether or not the null hypothesis is rejected. This results from the formulation of the null hypothesis being the status quo. If we decide to reject the null hypothesis, then the findings from the experiment as a whole is called **significant**.

The decision whether to reject the null hypothesis is based on a**test statistic**. The test statistic is a function of the random variables modelling the data. Hence it is a random variable itself, and its distribution depends on the parameters defining the model. For any specific hypothesis test, the test statistic that we choose needs to distinguish between the null and the alternative hypotheses, and have a distribution that is known and computable.

In the mammography example, we choose the test statistic $T = Y = X_1 + X_2 + ... + X_31000$, the number of deaths due to breast cancer in the treatment group. We expect $Y$ to take a smaller value under the alternative hypothesis $H_A: \pi < 0.00203$ than the null hypothesis $H_0: \pi = 0.00203$, so it will allow us to distinguish between the two. We also know the distribution of $Y$.

## 1.4.4 Significance level

We will use the distribution of the test statistic under the null hypothesis to answer the following question: "Assuming that the null hypothesis is true, how likely is it for the test statistic to be at least as extreme (in the direction of the alternative hypothesis) as the one we have computed?". To answer this question in the mammography example, we compute the probability of the test statistic $Y$ taking a more extreme value than the observed value 39, under the null $H_0: \pi = 0.00203$. Under the Poisson approximation, this is the probability of $Y < 39$ under the model $Y \sim Poisson(63)$.

In the general hypothesis testing framework, we will then reject the null hypothesis if this probability is small enough, since this implies that if the null hypothesis is true, a test statistic as extreme as what we had observed is very unlikely. The **significance level** $\alpha$ refers to this probability threshold. In many applications, $\alpha = 0.05$ is used.

<center><img src="Images\BinomialHypothesis.png" alt="Alt Text" width="500" height="400"></center>

**Hypothesis Testing Practice**

Suppose we now instead have 3000 participants in the treatment group and 31000 participants in the control group. As before, we base the null hypothesis on the breast cancer death rate observed in the control group, which is 0.00203, and define $Y$ as the number of deaths in the treatment group. Given that we now have different sized treatment and control groups, can we still model $Y$ as a Poisson variable (approximated from a Binomial) and perform the same hypothesis test using $Y$ as the test statistic? If so, what is the null hypothesis in terms of the parameter $\lambda$ (of the Poisson model)?

**Resp: Yes, $Y$ can still be modeled as Poisson($\lambda$), and the null hypothesis will be $H_0: \lambda = 60.9$**

Recall that we have the model $X_1, X_2, ..., X_31000 \space\space (i.i.d.) \space\space \sim \space\space Bernoulli(\pi)$ for the outcomes for each patient in the treatment group. We wish to distinguish between $H_0: \pi = 0.00203$ and $H_A: \pi < 0.00203$ using the test statistic $T$. How does $T$ distinguish between $H_0$ and $H_A$?

**Resp: If $T$ is large, then the data is more compatible to $H_0$, and if $T$ is small, especially much smaller than $0.00203*31000$, then the data is compatible to $H_A$.**

What is the distribution of $T$ under $H_0$?

**Resp: Binomial(31000, 0.00203)**

A hypothesis test is based on a given significance level $\alpha$. Following convention, we set $\alpha = 0.05$. For what observed value of $T$ should we reject $H_0$?

**Resp: For $T < 50$**

In the data collected, what is the observed test statistic $T$? From this, do we reject the null hypothesis at a significance level of 0.05?

**Resp: For $T = 39$. Reject the null hypothesis.**

## 1.4.5 Significance level and the p-value

Recall that hypothesis testing is based on the probability that the test statistic takes on its current value or a more extreme one, assuming that the null hypothesis $H_0$ is true. This probability is defined as the **p-value**. From the definition of the significance level, another intrepretation is that the -value is the smallest significance level $\alpha$ such that we will reject the null hypothesis. In other words, the p-value measures the "compatibility" of the observed data with the null hypothesis. The lower the P value, the less likely such data will be observed given the null hypothesis. Thus, the null hypothesis will be less convincing compared to the alternative hypothesis and we are more likely to reject it. The p-value can be calculated by summing up or integrating values along the tail of the test statistic pmf/pdf towards the direction of the alternative hypothesis.

<center><img src="Images\BinomialPValue.png" alt="Alt Text" width="450" height="400"></center>

The p-value is always between 0 and 1 and depends on the model. In the mammography study, the binomial model gives a slightly higher p-value (0.0012) than when the Poisson approximation is used (0.0008). In the exercise below, we will apply the concept of the p-value to the mammography study.

**P-value Practice**

Which of the following correctly describes the formula for the p-value in the mammography study, given an observation of $Y = 39$ deaths due to breast cancer in the treatment group of 31000?

**Resp: $P_{H_0} (T \le 39)$, the probability under $H_0$ to obtain the observed value or a more extreme value of the test statistic**

Suppose that we are instead interested in whether offering mammography makes a difference in the breast cancer death rate. We thus change the alternative hypothesis to $H_A: \pi \neq 0.00203$ while maintaining the null hypothesis $H_0: \pi = 0.00203$. Let $Y$ be the observed number of deaths in treatment group. Which of the following correctly represents the p-value of an observation, in terms of the observation $Y$?

**Resp: $P_{H_0} (|Y - 63| \ge 24)$, the probability under $H_0$ to obtain the observed value or a more extreme value of the test statistic**

## 1.4.6 Type I error and type II error

Hypothesis testing is an uncertain process due to inherent variation in the observations. There is thus the possibility of having the wrong conclusion, called **error**. The two types of error are as follows:

- **Type I error (false positive)**: We reject $H_0$ (equivalently, find the result significant) when $H_0$ is actually true. In the mammography study, it is concluding that offering mammography decreases the likelihood of breast cancer death even if it actually does not.
- **Type II error (false negative)**: We do not reject $H_0$ (equivalent, find the result not significant) when $H_A$ is actually true. In the mammography study, it is not concluding that offering mammography decreases the likelihood of breast cancer death when it actually does.

The two types of errors are shown in the off-diagonal entries in the matrix below. Note that the diagonal entries refer to a correct conclusion of either correctly rejecting or failing to reject the null hypothesis $H_0$ in favor of the alternative hypothesis $H_A$.

<center><img src="Images\ConfusionMatrix.png" alt="Alt Text" width="400" height="100"></center>

## 1.4.7 Power of a test

A related concept is the **power of a test**, which is defined as the probability of rejecting $H_0$ when $H_A$ is true, i.e. the probability of correctly rejecting the null hypothesis. In other words, **Power of test = 1 - P(Type II error)**.

The simplest case to discuss the power of a test is when we have a **simple hypothesis test**, which is when the test is parametric and both the null and the alternative hypotheses consist of a single parameter value. In the mammography study, we can for example have the following hypotheses on the parameter $\pi$.

- Null Hypothesis $H_0$: $\pi = 0.002$ (blue distribution on the right)
- Alternative Hypothesis $H_A$: $\pi = 0.0013$ (orange distribution on the left)

The distribution of the test statistic under the null and the alternative hypotheses are shown in the plot below. The blue distribution on the right corresponds to the null hypothesis $H_0$, while the orange distribution on the left corresponds to the alternative hypothesis $H_A$.

<center><img src="Images\H0HADistributions.png" alt="Alt Text" width="450" height="400"></center>

We reject $H_0$ if our sample has a p-value less than $\alpha$, i.e. any value on the x-axis to the left of the vertical line separating the colored from the non-colored regions. The power is then the area to the left of this line that is under the curve given $H_A$ (the left curve), i.e. the area of the orange and blue regions combined.

We note that:

- The <mark>significance level $\alpha$ is in fact the probability of a type I error</mark>. Thus, setting the significance level to $\alpha$ is equivalent to allowing this level of type I error to occur.
- The <mark> power of a test is in fact the probability of a type II error</mark>.

**Power of a test practice**

<left><img src="Images\EX1471.png" alt="Alt Text" width="650" height="200"></left>

<left><img src="Images\EX1472.png" alt="Alt Text" width="650" height="200"></left>

<left><img src="Images\EX1473.png" alt="Alt Text" width="650" height="300"></left>


## 1.4.8 Hypothesis testing trade-offs

There is a trade-off faced in hypothesis testing. In summary,

- There is a direct tradeoff between reducing the type I and type II errors. In nearly all cases, with a fixed test statistic, reducing the type I error results into increasing the type II error, and vice versa. It is possible to ilustrate this tradeoff looking at the image bellow: as you move the green vertical line defined by the $\alpha$ value to the left you reduce the P(type I error) but you increase the P(type II error). Similarly, if you move the vertical green line to the right you increase the P(type I error) at the same time you decrease the P(type II error).

- Keeping the significance level constant, a <mark>one-sided hypothesis test has a higher power than the corresponding two-sided hypothesis test. An exception is when the distribution of the observation under the alternative hypothesis is bimodal</mark>.

<center><img src="Images\H0HADistributions.png" alt="Alt Text" width="450" height="400"></center>


## 1.5 Different hypothesis Test for the mammography experiment

In the last lecture, we covered the basics of hypothesis testing with the HIP mammography study as our example. The study's aim is to determine whether offering mammographies for breast cancer detection reduces the rate of death due to breast cancer. There are 31000 individuals in each of the treatment and control groups; only those in the treatment groups are offered mammographies. We recap the elements of the hypothesis testing framework. In the mammography study, they are:

- The **Parametric Model**. We can write indicator variables for whether each patient in the treatment group dies of breast cancer as $X_1, X_2, ..., X_31000 \space\space (i.i.d.) \space\space \sim \space\space Bernoulli(\pi)$, and we can also approximate the total number of deaths as $Y = X_1 + X_2 + ... + X_31000 \space\space (i.i.d.) \space\space \sim \space\space Poisson(\lambda)$.

- The **null hypothesis** $H_0$: $\pi = 0.00203$ (equivalently $\lambda = 63$), and the **alternative hypothesis** $H_A$: $\pi < 0.00203$ (equivalently $\lambda < 63$). We then decide whether or not to reject the null hypothesis based on a test.

- The **test statistic** $T$. We define $T$ to simply be the number of deaths $Y$ in the treatment group. Under $H_0$, it is distributed as $T \sim \space\space Bernoulli(31000, 0.00203)$. This distribution can also be approximated as $T \sim \space\space Poisson(63)$. The role of $T$ is to distinguish between $H_0$ and $H_A$.

- The **significance level** $\alpha = 0.05$. This is the probability of rejecting the null hypothesis $H_0$ when it is in fact true (type I error), that is, the probability of concluding there is an effect when there is none. Generally, the threshold of the test statistic for rejecting the null hypothesis is set based on a chosen significance level.

- The **p-value** $p$. This is the probability that the test statistic, under the null hypothesis, takes a value more extreme (towards the direction of the alternative hypothesis) than the one observed. This probability can be computed from the test statistic $T$ and the given parametric model. The p-value varies with the observed value of data, and when $p < \alpha$, the $H_0$ is rejected.

- The **power of the test**. This is the probability of rejecting $H_0$ when $H_A$ is true (avoiding a type II error: 1 - P(Type II error)). It is useful to write the power as a function of the parameter, when more than one parameter value is considered for $H_A$.

- Throughout the hypothesis test, we focused on the observed death rate in the treatment group as the variable, and compare it to $\pi = 0.00203$, the observed death rate in the control group. The question below examines the validity of this approach.

**Exercise**

<left><img src="Images\EX1474.png" alt="Alt Text" width="650" height="300"></left>

## 1.5 Hypergeometric probability distribution

The hypergeometric distribution is a discrete distribution based on the following probability problem: "Suppose there are $N$ balls in a bowl, $K$ of which are red and the remaining $N - K$ of which are blue. From the bowl, $n$ balls are drawn without replacement. What is the probability that among the $n$ balls drawn, exactly $x$ are red?"

The solution to this problem is given by the following pmf:

$$
P(X = x) = \frac{\binom{K}{x}\binom{N - K}{n - x}}{\binom{N}{n}}
$$

This pmf defines the hypergeometric distribution **Hypergeometric**($N$, $K$, $n$) with the three parameters where,

 - $N$ is the size of population (number of balls in bowl);
 - $K$ is the size of sub-population of interest (number of red balls in bowl);
 - $n$ is the number of targeted outcomes (total number of balls drawn);
 - $\binom{K}{x}$ is the number of ways to choose $x$ out of $K$ red balls;
 - $\binom{N - K}{n - x}$ is the number of ways to choose $n - x$ out of $N - K$ blue balls;
 - $\binom{N}{n}$ is the number of ways ti choose $n$ balls out of $N$.


**Mammography study: Modified Hypotheses and Hypergeometric Distribution**. In the mammography study, another approach to test for treatment effect is to compare the numbers of breast cancer deaths in treatment and control groups on equal footing (instead of using the estimate of the control group as the status quo as before). That is, we now state the null hypothesis to be the death rates in the two groups are the same and the alternative hypothesis to be the death rate in the treatment is smaller:

- Null hypothesis $H_0$: $\pi_{treatment} = \pi_{control}$
- Alternative hypothesis $H_A$: $\pi_{treatment} < \pi_{control}$ 

where $\pi_{treatment}$ and $\pi_{control}$ are the death rates in the treatment and control groups respectively.


**Revised Hypothesis for the Mammography Study**

Remember that the role of a test statistic $T$ is to differentiate between $H_0$ and $H_A$, based on the observations. Furthermore, the distribution of $T$ needs to be known under the null hypothesis $H_0$. Recall the modified hypotheses above for the mammography study:

- Null hypothesis $H_0$: $\pi_{treatment} = \pi_{control}$
- Alternative hypothesis $H_A$: $\pi_{treatment} < \pi_{control}$ 

where $\pi_{treatment}$ and $\pi_{control}$ are the death rates in the treatment and control groups respectively.

<left><img src="Images\EX1475.png" alt="Alt Text" width="650" height="250"></left>

<left><img src="Images\EX1476.png" alt="Alt Text" width="650" height="250"></left>

<left><img src="Images\EX1477.png" alt="Alt Text" width="650" height="300"></left>


## 1.6 Fisher's Exact Test p-value

Fisher's Exact Test provides a method based on the hypergeometric distribution to test hypotheses of the form:

- $H_0$: $\pi_{treatment} = \pi_{control}$, i.e. treatment has no effect on the rate of occurence of a targeted outcome

- $H_A$: $\pi_{treatment} < \pi_{control}$, i.e. treatment lowers (or raises, or changes) the rate of occurence of a targeted outcome.

We define the test statistic $T$ to be the **number of targeted outcomes in the treatment group**. Under the null hypothesis that the treatment has no effect, $T$ follows a hypergeometric distribution **Hypergeometric**($N$, $K$, $n$), with parameters:

 - $N$ is the size of the experiment, i. e. total number of individuals in both treatment and control groups;
 - $K$ is the size of the treatment group;
 - $n$ is the number of targeted outcomes;

Recall the p-value is defined to be the probability that we obtain an observation as extreme or more extreme than the one observed,in the direction of the alternative hypothesis, under the null hypothesis. In a Fisher's exact test, this corresponds to the probability under a tail of the hypergeometric pmf.

**Application to the mammography study**: In the mammography study, the hypotheses of our Fisher's exact test is:

- Null hypothesis $H_0$: $\pi_{treatment} = \pi_{control}$, i.e. Treatment has no effect to the death rate due to breast cancer;
- Alternative hypothesis $H_A$: $\pi_{treatment} < \pi_{control}$, i.e. Treatment lowers the death rate due to breast cancer.

The test statistic $T$ for this test is the number of breast cancer deaths in the treatment group, which is distributed as hypergeometric with parameters $N = 62000$, $K = 31000$ and $n = 102$, as we discussed on the previous page. The p-value is then the sum of probabilities of obtaining a value of $T$ that is more extreme than $39$, in the direction of the alternate hypothesis. That is,

$$
p = P_{H_0}(T \leq 39) = \sum_{t=0}^{39}  \frac{\binom{31000}{t}\binom{31000}{102 - t}}{\binom{62000}{102}}
$$

From this, based on the significance level $\alpha$, we can either:

- reject the null hypothesis if $p \leq \alpha$, or
- fail to reject the null hypothesis if $p \geq \alpha$.

**Contingency Table**

Data for Fisher's exact test can typically be presented in a **contingency table**, which shows how the targeted outcomes are divided between the treatment and control group, as well as the sizes of these groups. In the mammography study, the contingency table looks like the following.

<center><img src="Images\ContingencyTable.png" alt="Alt Text" width="300" height="100"></center>

**Fisher's exact test practice**

<left><img src="Images\EX1478.png" alt="Alt Text" width="650" height="250"></left>

```python
#Importing libraries
import numpy as np
from scipy.stats import fisher_exact

# Creating a contingency table
contingency_table = np.array([[39, 63], [30961, 30937]])

# Calculating Fisher's exact test
statistic, pvalue = fisher_exact(contingency_table, alternative='less')

# Printing result
print(f"Statistics - prior odds ratio:{statistic} | p-value: {pvalue}")

Statistics - prior odds ratio:0.6185677526719483 | p-value: 0.011094091844052025
```

## 1.7 Paired tests and continuous data

The previous example had discrete observations, as our smallest relevant unit of observation is a binary variable that indicates whether a particular patient died of breast cancer. In many applications, however, the variable of interest may take on a values from a subset of $\mathbb{R}$ instead (such as an interval). We will examine one such study, which is evaluating the efficacy of a new sleeping drug.

**Sleeping drug study**: Suppose that a drug company is developing a new sleeping aid drug, which purportedly helps users suffering from insomia to increase their sleeping time each night. Validating whether this drug is effective is a hypothesis testing problem. Recall the following important elements of an experimental design:

- Control and treatment groups: There has to be a baseline for comparison.
- Randomization: The treatment assignment has to be random to avoid biases due to differing baseline average values in the two groups.
- Double-blindness: Patients should not know whether they received the placebo (control) or the actual drug (treatment) since knowing this may influence their behavior. Similarly the experimenters should not know who received treatment.

Based on the above, a basic idea would be to adapt a similar framework as the mammography study. We can take a large sample, split them into a treatment group, who will get the actual drug, and a control group, who will receive a placebo. We could then observe the number of hours slept in each group, set up a probability model, and continue the hypothesis testing procedure. <mark>However, even though this approach is valid, the power of this hypothesis test might not be very high</mark>, especially if the sample size is small. Indeed, people have a wide range of sleep lengths, and it might be difficult to discern anything due to this noise. An approach to reduce this noise is through a paired test design.

**Paired test design**. A paired test design involves taking multiple samples from an individual, one corresponding to a control situation and the other to a treatment situation. This allows us to estimate the effect on a particular individual. In the sleeping drug study, we want to observe the effect of taking the sleeping drug towards hours slept at night. In a paired test, it is the difference between the observed values in the treatment and the control **situations**, i.e. $Y_i := X_{i,treatment} - X_{i,control}$ that will be considered. A null hypothesis that states that the treatment has no effect is equivalent to claiming that $\mathbb{E}[Y_i] = 0$.

**Randomization and double-blindness in a paired test**. The usage of a paired test design removes the need for randomness in the treatment assignment, as each individual is observed for both the control and the treatment settings. However, randomization is still used to ensure that the individuals do not know whether they are in the treatment or the control group in each of the two trials, the double-blindness in the experiment.

In the sleeping drug study, this can be done by having two separate observation periods for each individual. Each individual is given the placebo in one period and the actual drug in the other, the assignment of which between the two is done at random. Also, having sufficient time between the two periods will prevent spillover effects of the drug from the first to the second.

A paired study like this cannot always be done, as it requires the possibility of having two distinct, independent observations and treatments for each indivdual. For example, in the mammography study we discussed, the effects of early breast cancer screening is lifelong, so we cannot measure or estimate the effect on a particular individual of having a mammography.

## 1.8 Z-test

In this section, we will discuss a popular and versatile approach to hypothesis testing on **continuous data**, the **z-test**, which makes use of the **Central Limit Theorem (CLT)**. We will apply this test to the sleeping drug study. Afterwards, we will see how the <mark>z-test is also helpful as an approximation when the data is discrete</mark>, such as in the mammography study.

<mark>When our data was binary, we are typically limited to the Bernoulli model and the corresponding binomial model for the number of targeted observations</mark>. When our data can take on continuous values, we have more choices. Depending on the application, we can use one of several well-known distributions, including the uniform, exponential, and normal distributions. Recall the data collected for the sleeping drug study:

<center><img src="Images\DrugTest.png" alt="Alt Text" width="500" height="100"></center>

Suppose our candidate models for the difference in number of hours slept are the uniform and the Gaussian models. Both the support and the distribution are important considerations:

- <mark>The support of a model is the set of values that the observations can take in the model</mark>. In the sleeping drug study, the number of hours slept in a day is bounded above, so the difference is also bounded. This points in favor of the uniform model, as it has a bounded support, while a Gaussian model always has unbounded support.
- The distribution of a continuous model is based on the shape of the pdf (probability distribution function). In model selection, this can be decided based on solving a theoretical model, looking at the empirical distribution of observations, or common knowledge. The number of hours slept by an adult is known to be centered around 8 hours, and outliers tend to be rare, so this points towards the Gaussian model for the sleeping drug study.

Weighing these two considerations, in the sleeping drug study, we select the normal distribution and then ensure that the variance parameter is sufficiently small, so that the probability of falling outside the realistic boundary is negligible. Furthermore, we can argue towards a normal distribution by reasoning that the number of hours slept is a cumulative effect of a large number of biological and lifestyle variables. As a lot of these variables are unrelated to one another, the cumulative effect can be approximated by a normal distribution. This is justified by the Central Limit Theorem (CLT), which is covered in more detail below, and is the important result that establishes the z-test.

## 1.8.1 Central limit theorem (CLT) and the z-test statistic

Suppose that we have observations $X_1,..., X_n$, which are independent and identically distributed based on a probability model. Under a few regularity assumptions (such as the model having a finite second moment), the distribution of the sample mean $\bar{X}$ will approximate a normal distribution when sample size becomes sufficiently large (typically $n \geq 30$).

The **central limit theorem (CLT)** states that: When sampling random variables $X_1,..., X_n$ from a population with mean $\mu$ and variance $\sigma^{2}$, $\bar{X}$ is approximately normally distributed with mean $\mu$ and variance $\frac{\sigma^{2}}{n}$ when $n$ is large:

$$
\bar{X} := \frac{X_1 + X_2 + ... + X_n}{n} \sim \Nu \left(\mu,  \frac{\sigma^{2}}{n}\right), \space\space for \space\space n \space\space large
$$

Hence, we can define a test statistic $z = \frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}}$ , which approximately follows a standard normal distribution when $n$ is large:

$$
z = \frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}} \sim \Nu \left(0,  1\right)
$$

The test statistic $z$ is called an (approximate) **pivotal quantity**, since its (approximate) distribution does not depend on the paramaters $\mu$ or $\sigma$. We can use the cdf of a pivotal quantity to compute the p-value (which is the probability for the test statistic to take on a value at least as extreme as the one observed), and compare the p-value with $\alpha$ the significance level to decide whether to reject the null hypothesis $H_0$.

We are interested in testing the efficacy of a sleeping drug. The data collection process recorded the hours of sleep of 10 patients under the drug and under the placebo:

<center><img src="Images\DrugTest.png" alt="Alt Text" width="500" height="100"></center>

Now, we want to answer the question: "Does the drug increase hours of sleep enough to matter?". We model the difference of hours of sleep between the drug and the placebo for each patient as a normal random variable:

$$
Model: \space X_1,..., X_10 \sim \Nu\left(\mu,  \sigma^{2}\right) \space (X_1, \space for \space example, \space would \space be: 6.1 - 5.2 = 0.9).
$$

From this, we state the hypotheses for a one-sided test:

- Null hypothesis ($H_0$): $\mu = 0$;
- Alternative hypothesis ($H_A$): $\mu > 0$

Since the data $X_i$ are modeled as independent Gaussians, the z-test statistic described above has an standard normal distribution under the null hypothesis $H_0$, even without using the central limit theorem. We consider using $z$ as the test statistic. However, to calculate $z = \frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}}$, we need to know the true value of the variance $\sigma$. Since we do not know the population variance in this experiment, we cannot use the z-test.

<mark>In general, if samples cannot be modeled as Gaussian variables, then the sample size also needs to be large in order to use the standard normal to approximate z using the CLT. The t-test resolves both issues of the unknown true variance and the required large sample size</mark>.

**Application to the mammography study**

We conduct the z-test for the mammography study with the following model and hypotheses:

- Model: $X_1, X_2, ..., X_31000 \space\space (i.i.d.) \space\space \sim \space\space Bernoulli(\pi)$ each indicating whether a patient in the treatment group dies of breast cancer.

- Null hypothesis $H_0$: $\pi = 63/31000$; Alternative hypothesis $H_A$: $\pi < 63/31000$.

As done in lecture 1, we have assumed in the null hypothesis that $\pi = 63/31000 \approx 0.00203$ is the true reference value for the death rate without treatment. Hence, we will assume the true variance of $X$ to be the corresponding value $\sigma = \sqrt{\pi (1 - \pi)} \approx 0.045$. The z-test statistic is:

$$
z = \frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}} = \frac{\frac{39}{31000}-\frac{63}{31000}}{\frac{\sqrt{(\frac{63}{31000})(1- \frac{63}{31000})}}{\sqrt{31000}}} \approx -3.0268
$$

The p-value can be calculated from the area under the pdf of the standard normal distribution to the left of the z-value:

<center><img src="Images\zGaussian.png" alt="Alt Text" width="300" height="300"></center>

**Z-test Practice**

<left><img src="Images\EX1479.png" alt="Alt Text" width="650" height="350"></left>

```python
#Importing libraries
import numpy as np
from scipy.stats import norm

# Calculating probability of death in the control group
prob_c = 63/31000

# Calculating probability of death in the treatment group
prob_t = 39/31000

# Defining the sample size n
n = 31000

# Defining the standard deviation
sigma = np.sqrt(prob_c*(1-prob_c))

# Calculating z
z = (prob_t - prob_c)/(sigma/np.sqrt(n))

# Calculating the cumulative distribution function from -inf to z considering that
# z is a variable sampled from a normal distribution with mean 0 and variance 1 
pvalue = norm.cdf(z, loc=0, scale=1)

# Printing result
print(f"z statistic:{z} | p-value: {pvalue}")

z statistic:-3.0267929604748476 | p-value: 0.0012358159238705215
```

## 1.9 T-test

In the previous example, we were able to define a pivotal statistic based on standardizing the sample mean. This, however, requires knowing the population variance, which is not the case. The t-test introduced following is a solution to this by estimating the unknown variance. <mark>Note that the t-test method can only be applied when we have a Gaussian model.</mark>

<mark>The t-test is a statistical method to test a hypothesis without knowing the population standard deviation $\sigma$</mark>. Instead, we can estimate $\sigma$ using the sample standard deviation formula, based on the observations $X_1, X_2,...,X_n$, where $\bar{X}$ is the sample mean. The formula is given by:

$$
\hat{\sigma} = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})^{2}}
$$

Note that in the above formula, we use a denominator $n-1$ instead of $n$, which is called *Bessel's correction* and makes the *sample variance unbiased*.

The t-statistic is computed similarly to the z-statistic, except that for the t-statistic, we substitute a known population variance $\sigma$, which does not exist in this setting with the sample variance $\hat{\sigma}^{2}$. The t-statistic is defined as:

$$
T = \frac{\bar{X} - \mu}{\frac{\hat{\sigma}}{\sqrt{n}}}
$$

Under the assumption that $X_1, X_2,...,X_n \space\space (i.i.d.) \sim \space \Nu(\mu, \sigma^{2})$ for any pair of parameters $(\mu, \sigma^{2})$, $T$ is a pivotal statistic. Its distribution is called a t-distribution and is parameterized by the number of **degrees of freedom**. In this case, , the  distribution with <mark> $n-1$ degrees of freedom </mark>. For applications, the $t$ distribution is well-known and can be accessed by standard tables or by using software packages. We will discuss the t-distribution further in the next sessions; for now, we will apply this to the sleeping drug study.

**Application to the sleeping drug experiment**

Continuing with sleeping drug experiment, we model the difference of hours of sleep between drug and placebo for each patient using the model $X_1, X_2,...,X_n \space\space (i.i.d.) \sim \space \Nu(\mu, \sigma^{2})$. The differences are shown in the table below:

<center><img src="Images\DrugTest2.png" alt="Alt Text" width="400" height="50"></center>

Then, we have that under the null hypothesis $H_0$, which is that $\mu = 0$, the test statistic $T$ follows the t-distribution with $n-1 = 9$ degrees of freedom:

$$
t_9 \sim \frac{\bar{X}}{\frac{\hat{\sigma}}{\sqrt{n}}}
$$

The pdf of the $t_9$ distribution, the distribution of the test statistic, is shown below and is compared with $\Nu(0,1)$.

<center><img src="Images\t9.png" alt="Alt Text" width="350" height="300"></center>

<mark>A key feature of the $t_n$ distribution is based on the fact that estimating $\sigma$ introduces uncertainty if there are only a few samples. Thus, the smaller degrees of freedom (corresponding to smaller $n$), the more weight placed on the tails</mark>. When the degrees of freedom increases, to say beyond 30, the $t$ distribution in fact approaches standard normal distribution $\Nu(0,1)$.

<left><img src="Images\EX1480.png" alt="Alt Text" width="650" height="250"></left>

<left><img src="Images\EX1481.png" alt="Alt Text" width="650" height="120"></left>

```python
#Importing libraries
import numpy as np
from scipy.stats import t, ttest_1samp

# Defining the sample data
x = np.array([0.9, -0.9, 4.3, 2.9, 1.2, 3.0, 2.7, 0.6, 3.6, -0.5])

# Calculating probability of death in the treatment group
x_bar = x.mean()

# Defining the sample size n
n = x.shape[0]

# Defining the standard deviation
sigma_hat = np.sqrt(np.sum((x - x_bar)**2)/(n-1))

# Defining mu
mu = 0

# Calculating t statistic for n-1 degrees of freedom
t_stat = (x_bar - mu)/(sigma_hat/(np.sqrt(n)))

# Calculating the cumulative distribution function from t to +inf considering that
# t is a variable sampled from a t distribution wiht mean 0 and standard deviation 1
pvalue = 1 - t.cdf(t_stat, df=n-1, loc=0, scale=1)

# Printing result
print(f"Xbar: {x_bar} | Sample size: {n} | Sigma hat: {sigma_hat} | t-stat: {t_stat} | p-value: {pvalue}")

: Xbar: 1.78 | Sample size: 10 | Sigma hat: 1.7681126158201075 | t-stat: 3.1835383022188735 | p-value: 0.0055606927492846125

# Using function ttest_1samp
t_stat, pvalue = ttest_1samp(a=x, popmean=mu, alternative='greater')

# Printing result
print(f"t-stat: {t_stat} | p-value: {pvalue}")

: t-stat: 3.1835383022188735 | p-value: 0.005560692749284678
```

<left><img src="Images\EX1482.png" alt="Alt Text" width="650" height="350"></left>

## 1.9.1 t-statistic and the t-distribution

In the previous section, we introduced the t-test, t-statistic, and the t-distribution. The setup is that we consider samples $X_1, X_2,...,X_n \space\space (i.i.d.) \sim \space \Nu(\mu, \sigma^{2})$ for some mean $\mu$ and variance $\sigma$. Define the test statistic:

$$
T_n = \frac{\bar{X_n} - \mu}{\frac{\hat{\sigma}}{\sqrt{n}}}
$$

where we have,

$$
\bar{X_n} = \frac{1}{n}\sum_{i=1}^{n}X_i
$$


$$
\hat{\sigma}^{2} = \frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X_n})^{2}
$$

The main result is that the $T_n$ has a t-distribution with $n-1$ degrees of freedom. We discuss this result further.


We start by defining the t distribution and its parameter $k$ which specifies the number of degrees of freedom. The t distribution with $n$ degrees of freedom is defined as the distribution of $\frac{Y}{\sqrt{\frac{Z}{n}}}$, where:

 - $Y \sim \Nu(0,1)$ is a standard normal distribution;
 - $Z \sim \Chi_n^{2}$ is a chi-squared distribution with $n$ degrees of freedom;
 - $Y$ and $Z$ are independent.

As $n$ increases, the distribution has thinner tails; more precisely, the variance of the $t_n$ distribution is $\frac{n}{n-2}$. The t distribution for different values of $n$ are plotted in the figure below.

<center><img src="Images\tDistribution.png" alt="Alt Text" width="450" height="400"></center>

Intuitively, we can see a rough correspondence from the definition of the t-statistic.

- The sample mean in the numerator of the t statistic is normally distributed, just as the $Y$ in the numerator of the  distribution is.
- The sample variance in the denominator of the t statistic is a sum of squares, which is similar to how the chi-squared distribution in the denominator of the t distribution is defined.

Next, we provide a formal proof that $T$ indeed follows a t distribution with $n-1$ degrees of freedom.

**Proof that the t statistic follows a t distribution**

To prove that the t statistic follows a t distribution, we specify $Y$ and $Z$ such that $T = \frac{Y}{\sqrt{\frac{Z}{n}}}$ and so that the three conditions for $Y$ and $Z$ given above are satisfied. We first construct $Y$, which must have a $\Nu(0,1)$ distribution. We already know that the z-statistic $z = \frac{\bar{X_n} - \mu}{\frac{\sigma}{\sqrt{n}}}$ has a standard normal distribution, so we can let $Y = \frac{\bar{X_n} - \mu}{\frac{\sigma}{\sqrt{n}}}$. Then, we can solve for $Z$ by equating the expressions for the t-statistic and the $t_{n-1}$ distribution:

$$
T = \frac{Y}{\sqrt{\frac{Z}{n-1}}} = \frac{\bar{X_n} - \mu}{\frac{\hat{\sigma}}{\sqrt{n}}}
$$

Hence, we derive the corresponding $Z$ as:

$$
\sqrt{\frac{Z}{n-1}} = \frac{Y\frac{\hat{\sigma}}{\sqrt{n}}}{\bar{X_n} - \mu} = \frac{\frac{\hat{\sigma}}{\sqrt{n}}}{\frac{\sigma}{\sqrt{n}}} \rarr Z = (n-1)\frac{\hat{\sigma}^2}{\sigma^2} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(X_i - \bar{X_n})^2
$$

Note that $Y$ only depends on $\bar{X_n}$. Hence, it suffices to show that:

- $\frac{1}{\sigma^2}\sum_{i=1}^{n}(X_i - \bar{X_n})^2$ has a $\Chi_{n-1}^2$ distribution;
- $\bar{X_n}$ and $\sum_{i=1}^{n}(X_i - \bar{X_n})^2$ are independent.

A popular approach to show both at the same time is to consider a related quantity which has distribution $\Chi_{n-1}^2$, as $\frac{X_i - \mu}{\sigma}$ has a $\Nu(0,1)$ distribution:

$$
W = \sum_{i=1}^{n} \left(\frac{X_i - \mu}{\sigma}\right)^2 \sim \Chi_{n-1}^2
$$

By some algebra manipulation, we can write:

$$
W = \sum_{i=1}^{n} \left(\frac{X_i - \mu}{\sigma}\right)^2 = \frac{1}{\sigma^2} \sum_{i=1}^{n}(X_i - \bar{X_n})^2 + \frac{n}{\sigma^2} \sum_{i=1}^{n}(\bar{X_n} - \mu)^2
$$

We now reason using multivariate Gaussians, as $X_1, X_2,...,X_n$ are i.i.d. Gaussians. Therefore, $\bar{X_n} \sim \Nu(\mu, \frac{\sigma^2}{n})$, so $\frac{n}{\sigma^2} (\bar{X_n} - \mu)^2 \sim \Chi_{n-1}^2$. More generally, we can construct variables out of linear combinations of $X_1, X_2,...,X_n$. If we have a pair of such variables, they will be jointly Gaussian so they are independent iff they have zero covariance.

We apply this technique to show that $X_i - \bar{X_n}$ and \bar{X_n} are independent. Indeed,

$$
Cov(X_i, \bar{X_n}) = Cov(X_i, \frac{X_i}{n}) = \frac{\sigma^2}{n}
$$

and

$$
Cov(\bar{X_n}, \bar{X_n}) = \sum_{i=1}^{n}Cov(\frac{X_i}{n}, \frac{X_i}{n}) = n \left(\frac{\sigma^2}{n^2}\right) = \frac{\sigma^2}{n}
$$

where $X_i$ and $X_j$ are independent for any $i \neq j$. Hence, we get that $Cov(X_i - \bar{X_n}, \bar{X_n}) = 0$, and so $X_i - \bar{X_n}$ and $\bar{X_n}$ are independent. Using the above fact for $i = 1,...,n$, this proves the claim that $\bar{X_n}$ and $\sum_{i=1}^{n}(X_i - \bar{X_n})^2$ are independent. Hence, the two components of W are also independent.

As the latter component has a $\Chi_1^2$ distribution, the former must have a $\Chi_{n-1}^2$ distribution. This is based on the additivity property of a $\Chi^2$ distribution: the sum of a $\Chi_1^2$ and $\Chi_{n-1}^2$ distribution, the two independent from each other, is a $\Chi_n^2$ distribution.

The uniqueness of this distribution can be shown by considering the uniqueness of the moment generating function. Indeed, write $W = W_1 + W_2$, where $W_1$ and $W_2$ are independent. Therefore, $M_w(t) = M_{w_1}(t)M_{w_2}(t)$, and obviously $M_{w_1}(t) = \frac{M_{w}(t)}{M_{w_2}(t)}$. From the mgf of $W_1$ we show that there is a unique corresponding distribution.

## 1.9.2 t-test and the normality assumption



**Exercise**

<left><img src="Images\EX1483.png" alt="Alt Text" width="650" height="350"></left>


## 1.10. Confidence interval

We again use the same setup with observations $X_1, X_2,...,X_n \space\space (i.i.d.) \sim \space \Nu(\mu, \sigma^{2})$. The model parameter $\mu$ is not known, but $\sigma^2$ is known (or we are able to assume a value for $\sigma^2$). Suppose that we are interested to infer the population mean  based on the observations and our model, but we are more interested in a range of realistic values. Such a range is called a **confidence interval**.

<mark>A confidence interval is an interval that is a function of the observations</mark>. It is closely related to the question of estimating the mean, as:

- it is **centered around the sample mean**;
- its width is **proportional to the standard error**.

The confidence interval is also parameterized by the significance level $\alpha$. The interval is defined so that with probability $1-\alpha$, the interval will contain the true mean $\mu$. This “probability" means that if we sample the dataset numerous times and calculate intervals for each time, the probability that $\mu$ is in the proposed range (resulting intervals) is $1-\alpha$. It could be defined as:

$I(x) = {\theta | H_0: \mu = \theta}$ Can not be rejected at significance level, given the data X.

Figure below shows the probability of z-test statistic can take between $-\Phi_{1-\frac{\alpha}{2}}^{-1}$ and $\Phi_{1-\frac{\alpha}{2}}^{-1}$, where $\Phi$ is the Cumulative Distribution Function of the standard normal distribution, $\alpha$ is the significant level.

<center><img src="Images\confidence_interval.png" alt="Alt Text" width="450" height="350"></center>

which means that:

$$
P(-\Phi_{1-\frac{\alpha}{2}}^{-1} \leq \frac{\bar{X_n} - \mu}{\frac{\sigma}{\sqrt{n}}} \leq \Phi_{1-\frac{\alpha}{2}}^{-1}) = 1 - \alpha
$$

$$
P(\bar{X}-\frac{\sigma}{\sqrt{n}}\Phi_{1-\frac{\alpha}{2}}^{-1} \leq \mu \leq \bar{X}+\frac{\sigma}{\sqrt{n}}\Phi_{1-\frac{\alpha}{2}}^{-1})
$$

Hence, the confidence interval for true parameter $\mu$ with probability $1 - \alpha$:

$$
\bar{X} \pm \frac{\sigma}{\sqrt{n}}\Phi_{1-\frac{\alpha}{2}}^{-1}
$$

The figure shows the confidence interval for 100 times simulation (sampling the dataset for 100 times). The probability where $\mu$ falls into interval section is $1-\alpha$. As you can see from the figure, $\mu$ (red vertical line) is in the range of intervals (black horizontal line) in most of cases.

<center><img src="Images\Sample100.png" alt="Alt Text" width="450" height="350"></center>


**Probability Review: Cumulative Distribution Function**

The cumulative distribution function (cdf) is the probability that variable X takes a value less than or equal to x.

$$
F_x(x) = P(X \leq x)
$$

The figure shows the cumulative distribution function for a standard normal distribution:

<center><img src="Images\cdf.png" alt="Alt Text" width="450" height="350"></center>

The shade area in following figure represents $\Phi_{1-\frac{\alpha}{2}}$, where $\alpha = 0.05$:

<center><img src="Images\shaded.png" alt="Alt Text" width="450" height="350"></center>


**Confidence Interval Exercise**

<left><img src="Images\EX1484.png" alt="Alt Text" width="650" height="150"></left>

## 1.11. Summary

In this lecture, we have discussed three different test statistics: Fisher Exact Test, Z - Test and T-Test. The table below provides a summary of properties of these three tests for reference.

<center><img src="Images\SummaryTable.png" alt="Alt Text" width="650" height="150"></center>

We note that, the T-test only works when the sample is drawn from a normal distribution. When conducting hypothesis testing, it is important to consider your claim and experiment data to pick the test approach most fitted to your objective.

## 1.12 Likelihood Ratio Test and Multiple Hypothesis Testing

The likelihood ratio test can be applied to a general setting with:

- Hypotheses: $H_0: \theta \in \Theta_0, H_A: \theta \in \Theta_A$ where $\Theta_0$ and $\Theta_A$ are disjoint subsets $\left(\Theta_0 \cap \Theta_A = \emptyset\right)$ of a parameter space $\Theta  = \Theta_0 \cup \Theta_A$.
- Parametric model with parameter $\theta$: The probability (or likelihood) of observing the data $x$ is $p(x;\theta)$.with the parameter $\theta$.

The likelihood ratio test statistic $\Lambda(x)$ is defined as (negative twice) the logarithm of the likelihood ratio $L(x)$:

$$
\Lambda(x) = -2log(L(x))
$$

where,

$$
L(x) = \frac{max_{\theta \in \Theta_0} p(x; \theta)}{max_{\theta \in \Theta} p(x; \theta)}
$$
 				 	 
Equivalently, in the language of maximum likelihood estimators, we can write:

$$
L(x) = \frac{ p(x; \hat{\theta}_{MLE}^{constrained})}{p(x; \hat{\theta}_{MLE})}
$$

 				 	 
where $\hat{\theta}_{MLE}$ is the **maximum likelihood estimator** of $\theta$ and $\hat{\theta}_{MLE}^{constrained}$ is the constrained maximum likelihood estimator of $\theta$ within $\Theta_0$.

<br></br>

**Likelihood Ratio Test Intuition**

<left><img src="Images\EX1485.png" alt="Alt Text" width="650" height="200"></left>

## 1.12.1 The Distribution of Likelihood Ratio test statistics

**Wilk's Theorem** states that when the sample size is large, the distribution of $\Lambda$ under $H_0$ approaches a $\Chi^2$ distribution:

$$
\Lambda \rarr \Chi_{d}^{2}, \space as \space n \rarr \infty
$$

where $d$ is the **degree of freedom** of the $\Chi^{2}$ distribution and $d = dim(\Theta) - dim(\Theta_0)$.

**Power of Likelihood Ratio Test**

The **Neyman–Pearson lemma** states that among all tests that test for the simple hypotheses $H_0: \theta = \theta_0; H_A: \theta = \theta_A$ at significance level $\alpha$, the likelihood ratio test is the most powerful. That is, among all tests testing the same simple hypotheses and at the same significance level, the likelihood ratio test gives the largest probability of rejecting the null when indeed the alternate is true.

**Exercises**

<left><img src="Images\EX1486.png" alt="Alt Text" width="650" height="350"></left>

In the following problems, we apply the likelihood ratio test to the mammography study to answer the question whether the treatment changes the death rate. The corresponding hypotheses in terms of the breast cancer death rates, $\pi_T$ in the treatment group and $\pi_C$ in the control group, are:

$$
H_0: \pi_T = \pi_C
$$

$$
H_A: \pi_T \neq \pi_C
$$

<left><img src="Images\EX1487.png" alt="Alt Text" width="650" height="350"></left>

Let $Y_T$ and $Y_C$ be the numbers of cancer deaths in the treatment and control groups respectively. Assuming these are independent from each other, the probability of having $y_t$ breast cancer deaths in the treatment group and $y_c$ breast cancer deaths in the control group is the product:

$$
P(Y_T = y_t, Y_C = y_c) = P(Y_T = y_t)P(Y_C = y_c)
$$

Recall the HIP mammography study data:

<center><img src="Images\BreastCancer.png" alt="Alt Text" width="650" height="80"></center>
 				 	 
We use the binomial model for $Y_T$ and $Y_C$:

- $Y_T \sim Binom(31000, \pi_T)$
- $Y_C \sim Binom(31000, \pi_C)$

The likelihood ratio test statistic is:

$$
\Lambda(y_t, y_C) = -2log\left(\frac{max_{\Theta_0} P(y_t, y_C; \pi_T,\pi_C)}{max_{\Theta_A} P(y_t, y_C; \pi_T,\pi_C)}\right)
$$

$$
= -2log\left(\frac{max_{\pi_T = \pi_C \in [0,1]} P(y_t, y_C; \pi)}{max_{\pi_T \neq \pi_C} P(y_t, y_C; \pi_T,\pi_C)}\right)
$$

$$
= -2log\left(\frac{max_{\pi_T = \pi_C \in [0,1]} P(Binom(31000, \pi) = y_T)P(Binom(31000, \pi) = y_C)}{max_{\pi_T \neq \pi_C} P(Binom(31000, \pi_T) = y_T)P(Binom(31000, \pi_C) = y_C)}\right)
$$

$$
= -2log\left(\frac{max_{\pi_T = \pi_C \in [0,1]} P(Binom(31000, \hat{\pi}^{MLE}) = y_T)P(Binom(31000, \hat{\pi}^{MLE}) = y_C)}{max_{\pi_T \neq \pi_C} P(Binom(31000, \hat{\pi}_{T}^{MLE}) = y_T)P(Binom(31000, \hat{\pi}_{C}^{MLE}) = y_C)}\right)
$$

where we have used $P(Binom(n,p) = y)$ to denote the probability that a binomial variable with parameters $n,p$ takes value $y$.

Based on the observed data, Find the parameters $(\pi_T, \pi_C)$ that maximize the numerator and the denominator in the definition of the test statistic $\Lambda$. That is, find the 3 different maximum likelihood estimates in the expression above.

<center><img src="Images\MLE.png" alt="Alt Text" width="650" height="400"></center>

Similarly, for $P(k_1;n_1,p)P(k_2;n_2,p)$, we have:

$$
ln[P(k_1;n_1,p)P(k_2;n_2,p)] = ln \left[\binom{n_1}{k_1}p^{k_1}(1-p)^{n_1-k_1}\binom{n_2}{k_2}p^{k_2}(1-p)^{n_2-k_2}\right] = ln\binom{n_1}{k_1} + k_1ln(p) + (n_1-k_1)ln(1-p) + ln\binom{n_2}{k_2} + k_2ln(p) + (n_2-k_2)ln(1-p)
$$

Taking the partial derivative with respect to $p$:

$$
\frac{\partial(ln[P(k_1;n_1,p)P(k_2;n_2,p)])}{\partial p} = \frac{k_1}{p_1} - \frac{n_1 - k_1}{1 - p} + \frac{k_2}{p_2} - \frac{n_2 - k_2}{1 - p}
$$

$$
p = \frac{k_1 + k_2}{n_1 + n_2}
$$

```python

#Importing libraries
import numpy as np
from scipy.stats import chi2, binom

# Number of deaths in the treatment set
k_T = 39

# Number of deaths in the control set
k_C = 63

# Number of patients in the treatment set
n_T = 31000

# Number of patients in the control set
n_C = 31000

# Maximum likelihood estimator and Probability Mass Function for the treatment set under the alternative hypothesis (HA): pMLE_T diff pMLE_C
pMLE_T = k_T/n_T
pmf_T = binom.pmf(k=k_T, n=n_T, p=pMLE_T, loc=0)
print(f"Maximum likelihood estimator (MLE) treatment: {pMLE_T} and Probability Mass Function (PMF) treatment: {pmf_T}")

: Maximum likelihood estimator (MLE) treatment: 0.0012580645161290322 and Probability Mass Function (PMF) treatment: 0.06378567544145823

# Maximum likelihood estimator and Probability Mass Function for the control set under the alternative hypothesis (HA): pMLE_T diff pMLE_C
pMLE_C = k_C/n_C
pmf_C = binom.pmf(k=k_C, n=n_C, p=pMLE_C, loc=0)
print(f"Maximum likelihood estimator (MLE) treatment: {pMLE_C} and Probability Mass Function (PMF) treatment: {pmf_C}")

: Maximum likelihood estimator (MLE) treatment: 0.002032258064516129 and Probability Mass Function (PMF) treatment: 0.05024664607042506

# Maximum likelihood estimator and Probability Mass Function for the control / treatment set under the null hypothesis (H0): pMLE_T = pMLE_C = pMLE
pMLE = (k_T + k_C)/(n_T + n_C)
pmf_p = binom.pmf(k=k_T, n=n_T, p=pMLE, loc=0)*binom.pmf(k=k_C, n=n_C, p=pMLE, loc=0)
print(f"Maximum likelihood estimator (MLE) treatment: {pMLE} and Probability Mass Function (PMF) treatment: {pmf_p}")

: Maximum likelihood estimator (MLE) treatment: 0.0016451612903225807 and Probability Mass Function (PMF) treatment: 0.00018449866030670578

# Calculating the likelihood ratio test statistic
LR_stat = -2*np.log(pmf_p/(pmf_T*pmf_C))
print(f"The likelihood ratio test statistic is {LR_stat}")

: The likelihood ratio test statistic is 5.709660479762178

# Calculating p-value assuming a Chi-Squared distribution for LR_stat
pvalue = 1 - chi2.cdf(x=LR_stat, df=1, loc=0, scale=1)
print(f"The p-value is {pvalue}")

: The p-value is 0.016871802195942753
```

<left><img src="Images\EX1488.png" alt="Alt Text" width="650" height="350"></left>

<left><img src="Images\EX1489.png" alt="Alt Text" width="650" height="350"></left>

## 1.13 Metrics for false significance

When testing one hypothesis, the main error that we control is the error of making a false discovery, i.e. the type I error of rejecting the status quo even though it is true. We control this by setting the significance level $\alpha = P(reject \space H_0 \space | \space H_0 \space true)$ to be a small value (typically 0.05). When performing multiple hypothesis tests, setting the significance level for all tests to be some fixed $\alpha$ (independent of the number of tests) may not be enough to control false significance. For example, setting $\alpha = 0.05$ for each test means that the expected number of false discoveries is $5%$ of the total number of the tests for which the null hypotheses are true. So the number of false discoveries may be large, especially when a large number of tests are performed. Instead, in the setting of multiple testing, we can control the two following metrics for false significance:

- **Family-wise error rate (FWER)**: the probability of making at least one false discovery, or type I error;
- **False discovery rate (FDR)**: the expected fraction of false significance results among all significance results.

## 1.13.1 Family-wise error rate (FWER)

For a series of tests in which the i-th test uses a null hypothesis $H_{0}^{i}$, let the total number of each type of outcome be as follows:

<center><img src="Images\FWER.png" alt="Alt Text" width="600" height="100"></center>

Then the **family-wise error rate (FWER)** is the probability of making at least one false discovery, or type I error;

$$
FWER = P(V \geq 1)
$$

where $V$ is the total number of type I errors as in the table above, i.e., $V = \sum_{i=1}^{m_0} \psi_i$, where $\psi_i$ is the set of $m_0$ tests for which $H_0$ is true. In scenarios in which any false claims of discovery may lead to serious consequences, such as for drug approval, we want to control FWER.

**FWER with no corrections**

Recall from the lecture the paired test in which treatment effects are measured on 100 variables for 1000 people, and the treatment itself is a placebo (of being given water). If we perform $m$ independent tests each at significant level $\alpha$, then the FWER is:

$$
FWER = P(V \geq 1) = 1 - P(V = 0) = 1 -(1 - \alpha)^{m} \approx 1 \space for \space\space large \space\space m
$$

In other words, if we set the significance level of each test without taking into account the large number of tests performed, it is highly likely that the series of tests will lead to at least one false discovery. This often leads to puzzling claims such as water has treatment effect on important health parameters, or eating pizza reduces the risk of cancer.

We can control the size of FWER by choosing significance levels of the individual tests to vary with the size of the series of tests. In practice, this translates to correcting p-values before comparing with a fixed significance level e.g. $\alpha = 0.05$.

## 1.13.2 Bonferroni Correction

In a series of $m$ tests, if the significance level of each test is set to $\frac{\alpha}{m}$, or equivalently if the null hypothesis $H_0^{i}$ of each test $i$ is rejected when the corresponding p-value is bounded by:

$$
p^{i} < \frac{\alpha}{m}, (equivalently \space\space mp^{i} < \alpha)
$$

then $FWER < \alpha$. In other words, to achieve $FWER < \alpha$, we can “correct" the p-value $p_i$ of each test to $mp^{i}$ and reject $H_0^{i}$ when $mp^{i} < \alpha$. This correction is called the **Bonferroni correction**. The Bonferroni correction apples to a series of tests that are not necessary independent of each other. However, when the number of tests are large, the rejection criteria are stringent, and this may lead to low power of the tests.

<center><img src="Images\BonferroniProof.png" alt="Alt Text" width="600" height="500"></center>

## 1.13.3 Holm-Bonferroni Correction

The Holm-Bonferroni method makes adaptive adjustments to the rejection criterion of each test. The procedure is as follows. Suppose we are testing for $m$ hypotheses.

* Sort the $m$ p-values in increasing order $p_{(1)} \leq p_{(2)} \leq ...  \leq p_{(i)} \leq ... \leq p_{(m)}$.
* Start with $i = 1$, i.e. the minimum p-value. If

$$
p_{(i)} \leq \frac{\alpha}{m - (i - 1)}
$$

then, reject $H_0^{i}$ and proceed to the next smallest p-value, increment $i$ by 1, and again use the rejection criterion above.

* As soon as a hypothesis, say $H_0^{k}$, is not rejected, stop the procedure and simply do not reject any of $H_0^{k}, H_0^{k+1} ... H_0^{m}$.

This procedure guarantees $FWER < \alpha$ for the series of $m$ tests (which do not need to be independent). The adjustment from the p-value to $(m - (i - 1))p_{(i)}$ before comparing with $\alpha$ is known as the **Holm-Bonferroni** correction .

The <mark>Holm-Bonferroni method is more powerful than the Bonferroni correction</mark>, since it increases the chance of rejecting the null hypothesis (accepting discovery), and hence reduces the chance of making type II error.

**Exercises**

<left><img src="Images\EX1490.png" alt="Alt Text" width="650" height="200"></left>

<left><img src="Images\EX1491.png" alt="Alt Text" width="650" height="200"></left>

<left><img src="Images\EX1492.png" alt="Alt Text" width="650" height="300"></left>

## 1.14 False Discovery Rate (FDR) and Comparisons

Sometimes, controlling FWER (the probability of making one or more false discoveries) may be too strict for any discovery to be reported. Instead, we can then control the expected proportion of false discoveries among all discoveries made, the false discovery rate (FDR).

<center><img src="Images\FDR.png" alt="Alt Text" width="600" height="100"></center>

Recall $N_1$ is the total number of discoveries made (the total number of null hypotheses rejected), and $V$ is the number of false discoveries (the number of null hypotheses that were falsely rejected). Hence $V \leq N_1$ and $V / N_1$ is a ratio that is always between zero and one. If no null hypotheses were rejected, i.e. if $N_1 = 0$, we define the ratio $V / N_1$ to be zero to avoid a division by zero.

The **false discovery rate (FDR)** is:

$$
FDR = E \left[\frac{V}{N_1}\right]
$$

**Benjamini-Hochberg Correction**

The Benjamini-Hochberg method guarantees FDR $\leq \alpha$ for a series of $m$ independent tests. The procedure is as follows:

* Sort the $m$ p-values in increasing order $p_{(1)} \leq p_{(2)} \leq ...  \leq p_{(i)} \leq ... \leq p_{(m)}$.
* Find the maximum $k$ such that

$$
p_{(k)} \leq \frac{k}{m} \alpha
$$

* Reject all of  $H_{0}^{(1)}, H_{0}^{(2)},..., H_{0}^{(k)}$

Note that in this procedure, some null hypothesis $H_{0}^{(i)}$ can be rejected even if $p_{(i)} > \frac{i}{m} \alpha$ provided that $i < k$. The adjustment of the p-value to $\frac{m}{i}p(i)$ before comparing with $\alpha$ is called the **Benjamini-Hochberg correction**. For example, the table below shows the p-values from 5 hypothesis tests in an experiment in increasing order. We compute the adjusted p-value and compare it with significance threshold of 5%, to decide whether to reject the null hypothesis:

<center><img src="Images\FDRRank.png" alt="Alt Text" width="300" height="100"></center>

**FDR versus FWER**

<mark>Compared to FWER, FDR has higher power. Put another way, FWER is stricter than FDR<\mark>. Let us examine this by considering the trivial scenario where all null hypotheses are true. In this case, any rejected null hypothesis must also be falsely rejected, hence $V = N_1$. If any null hypothesis was rejected, then $V/N_1 = 1$, or if none was rejected, then $V/N_1 = 0$.

Recall that FWER is the probability that one or more null hypotheses were falsely rejected. In this scenario, this is the same as the probability that one or more null hypotheses were rejected, since any rejection is a false rejection. We can see now that if one or more null hypotheses were rejected, then $V/N_1 = 1$, and so:

$$
E \left[\frac{V}{N_1}\right] = P(V \geq 1)
$$

$$
FDR = FWER
$$

Now consider the general case when some null hypotheses may be false. This time, when $N_1 > 0$, we only know that $V/N_1 \leq 1$. Define an indicator variable $1P(V \geq 1)$ which takes value 1 when $V \geq 1$. Then:

$$
\frac{V}{N_1} \leq 1(V \geq 1) \space\space when \space\space V \geq 1
$$

$$
\frac{V}{N_1} = 1(V \geq 1) = 0 \space\space when \space\space V = 0
$$

Taking expectation on both sides, we see:

$$
E \left[\frac{V}{N_1}\right] \leq E\left[1(V \geq 1)\right] = P(V \geq 1)
$$

$$
FDR \leq FWER
$$

Since FDR is less than FWER, it is easier to control than FWER. That is, we do need to apply as large a correction factor for the FDR than we do to the FWER to get it under our significance threshold $\alpha$. Therefore, the power of a series of tests with FDR controlled will be larger that the power of the series with FWER controlled.

**Exercises**

<left><img src="Images\EX1493.png" alt="Alt Text" width="650" height="200"></left>

<left><img src="Images\EX1494.png" alt="Alt Text" width="650" height="200"></left>

<left><img src="Images\EX1495.png" alt="Alt Text" width="650" height="200"></left>

<left><img src="Images\EX1496.png" alt="Alt Text" width="650" height="150"></left>

## 2.1 Covariance

In statistics, covariance measures the joint variation between two random variables. For random variables $X$ and $Y$, the covariance is defined by:

$$
Cov(X,Y) = E\left[(X-\mu_{X})(Y-\mu_{Y})\right]
$$

where $\mu_{X}$, $\mu_{Y}$ are expectations of $X$ and $Y$, respectively.

Given a real data set ${(X_1, Y_1), (X_2, Y_2),..., (X_N, Y_N)}$, consisting of $N$ samples of pairs of random variables $X$ and $Y$, the covariance $Cov(X,Y)$ can be estimated by the empirical sum:

$$
Cov(X,Y) \approx \frac{1}{N} \sum_{i=1}^{N} (X_i-\bar{X})(Y_i-\bar{Y})
$$

where $\bar{X} = \frac{1}{N} \sum_{i=1}^{N} X_i$ and $\bar{Y} = \frac{1}{N} \sum_{i=1}^{N} Y_i$ are mean values. Similarly, the variance of random variables $X$ and $Y$ have empirical estimates given by:

$$
s_{X}^{2}  = E\left[(X-\bar{X})^{2}\right] \approx \frac{1}{N} \sum_{i=1}^{N} (X_i-\bar{X})^{2}
$$

$$
s_{Y}^{2}  = E\left[(Y-\bar{Y})^{2}\right] \approx \frac{1}{N} \sum_{i=1}^{N} (Y_i-\bar{Y})^{2}
$$

Note that the above estimates for corivance and variance are all biased, and an easy way to correct the bias is applying **Bessel's correction**, where we use $N-1$ to replace $N$ in all the denominators. In the rest of this lecture, both unbiased and biased estimates may appear.

In this exercise, we will investigate the correlation present in astronomical data observed by Edwin Hubble in the period surrounding 1930. Hubble was interested in the motion of distant galaxies. He recorded the apparent velocity of these galaxies – the speed at which they appear to be receding away from us – by observing the spectrum of light they emit, and the distortion thereof caused by their relative motion to us. He also determined the distance of these galaxies from our own by observing a certain kind of star known as a Cepheid variable which periodically pulses. The amount of light this kind of star emits is related to this pulsation, and so the distance to any star of this type can be determined by how bright or dim it appears.

The following figure shows his data. The Y-axis is the apparent velocity, measured in kilometers per second. Positive velocities are galaxies moving away from us, negative velocities are galaxies that are moving towards us. The X-axis is the distance of the galaxy from us, measured in mega-parsecs (Mpc); one parsec is 3.26 light-years, or 30.9 trillion kilometers.

<center><img src="Images\Hubble.png" alt="Alt Text" width="600" height="400"></center>

```python
# Creating Xs array
Xs = np.array([0.0339, 0.0423, 0.213, 0.257, 0.273, 0.273, 0.450, 0.503, 0.503, 0.637, 0.805, 0.904, 0.904, 0.910, 0.910, 1.02, 1.11, 1.11, 1.41, 1.72, 2.03, 2.02, 2.02, 2.02])

# Creating Ys array
Ys = np.array([-19.3, 30.4, 38.7, 5.52, -33.1, -77.3, 398.0, 406.0, 436.0, 320.0, 373.0, 93.9, 210.0, 423.0, 594.0, 829.0, 718.0, 561.0, 608.0, 1.04E3, 1.10E3, 840.0, 801.0, 519.0])

# Storing the number of points
n = Xs.shape[0]

print(f"Xs mean: {Xs.mean()} Mpc and Ys mean: {Ys.mean()} km/s")
:Xs mean: 0.9199250000000001 Mpc and Ys mean: 425.6175 km/s
print(f"Xs standard deviation: {np.sqrt(np.sum((Xs-Xs.mean())**2)/(n-1))} Mpc and Ys standard deviation: {np.sqrt(np.sum((Ys-Ys.mean())**2)/(n-1))} km/s")
:Xs standard deviation: 0.6533948258734996 Mpc and Ys standard deviation: 348.7336574977229 km/s
print(f"Covariance between Xs and Ys: {np.sum((Xs-Xs.mean())*(Ys-Ys.mean()))/(n-1)}")
:Covariance between Xs and Ys: 191.20706528260868
```

<left><img src="Images\EX1497.png" alt="Alt Text" width="650" height="250"></left>

<left><img src="Images\EX1498.png" alt="Alt Text" width="650" height="250"></left>

<left><img src="Images\EX1499.png" alt="Alt Text" width="650" height="400"></left>

<left><img src="Images\EX1500.png" alt="Alt Text" width="650" height="350"></left>

<left><img src="Images\EX1501.png" alt="Alt Text" width="650" height="250"></left>

<left><img src="Images\EX1502.png" alt="Alt Text" width="650" height="100"></left>

We notice that the units of the covariance make it awkward to interpret. It is clear that if the value is zero, there is no correlation, while a positive value shows some correlation, and a negative value is some anti-correlation. But exactly how much correlation or anti-correlation does the value you calculated entail? The number is not given in mega-parsecs, or kilometers per second, but something entirely different. The sample correlation coefficient addresses this problem by scaling the covariance by its maximum possible value:

<left><img src="Images\EX1503.png" alt="Alt Text" width="650" height="450"></left>

<left><img src="Images\EX1504.png" alt="Alt Text" width="650" height="350"></left>

<left><img src="Images\EX1505.png" alt="Alt Text" width="650" height="250"></left>

```python
# Creating Xs array
Xs = np.array([0.0339, 0.0423, 0.213, 0.257, 0.273, 0.273, 0.450, 0.503, 0.503, 0.637, 0.805, 0.904, 0.904, 0.910, 0.910, 1.02, 1.11, 1.11, 1.41, 1.72, 2.03, 2.02, 2.02, 2.02])

# Creating Ys array
Ys = np.array([-19.3, 30.4, 38.7, 5.52, -33.1, -77.3, 398.0, 406.0, 436.0, 320.0, 373.0, 93.9, 210.0, 423.0, 594.0, 829.0, 718.0, 561.0, 608.0, 1.04E3, 1.10E3, 840.0, 801.0, 519.0])

# Storing the number of points
n = Xs.shape[0]

# Calculating Xs mean
Xs_mu = Xs.mean()

# Calculating Ys mean
Ys_mu = Ys.mean()

# Calculating Xs standard deviation
Xs_std = np.sqrt(np.sum((Xs-Xs.mean())**2)/(n-1))

# Calculating Ys standard deviation
Ys_std = np.sqrt(np.sum((Ys-Ys.mean())**2)/(n-1))

# Standard Xs
Xs_z = (Xs - Xs_mu)/Xs_std

# Standard Ys
Ys_z = (Ys - Ys_mu)/Ys_std

# Covariance between Xs and Ys
CovXY = np.sum((Xs-Xs.mean())*(Ys-Ys.mean()))/(n-1)

# Correlation between Xs and Ys
rXY = np.sum(Xs_z*Ys_z)/(n-1)

results = f"""
Xs mean: {Xs_mu} Mpc
Ys mean: {Ys_mu} km/s
Xs standard deviation: {Xs_std} Mpc
Ys standard deviation: {Ys_std} km/s
Covariance between Xs and Ys: {CovXY} Mpc km/s
Correlation between Xs and Ys: {rXY} unitless
"""

print(results)

:Xs mean: 0.9199250000000001 Mpc
:Ys mean: 425.6175 km/s
:Xs standard deviation: 0.6533948258734996 Mpc
:Ys standard deviation: 348.7336574977229 km/s
:Covariance between Xs and Ys: 191.20706528260868 Mpc km/s
:Correlation between Xs and Ys: 0.8391399162310663 unitless
```

<left><img src="Images\EX1506.png" alt="Alt Text" width="650" height="100"></left>

<left><img src="Images\EX1507.png" alt="Alt Text" width="650" height="250"></left>

<left><img src="Images\EX1508.png" alt="Alt Text" width="650" height="200"></left>

<left><img src="Images\EX1509.png" alt="Alt Text" width="650" height="200"></left>

<left><img src="Images\EX1510.png" alt="Alt Text" width="650" height="200"></left>

## 2.2 Regression

By previous correlation analysis for the Hubbel dataset, we know there exists correlation relationship between the apparent velocity of a galaxy and its distance from us. Next, we want to model this relationship analytically. The most intuitive but prevalent model is the linear mapping:

$$
y = \beta_1 x + \beta_0 + \epsilon
$$

where $\beta_1$ is the slope of the linear model, $\beta_0$ is the model's intercept, and $\epsilon$ is the noise model.

To identify the model parameters $\beta_0$ and $\beta_1$, for a given dataset $\left[(X_i, Y_i)\right]_{i=1}^{N}$, we define the fitting error function

$$
E(\beta_0, \beta_1) = \sum_{i=1}^{N} (\beta_1 x + \beta_0 - Y_i)^{2}
$$

which is exactly the sum of all the squared residuals, i.e., the discrepancies between the predicted and observed velocities. Then the unknown parameters can be found by solving the following optimization problem:

$$
min_{\beta_0, \beta_1} E(\beta_0, \beta_1)
$$

As $E(\beta_0, \beta_1)$ is quadratic, we can show that the optimum is achieved when:

$$
\hat{\beta_0} = \bar{Y} - \hat{\beta_1} \bar{X}
$$

$$
\hat{\beta_1} = \frac{\sum_{i}^{N}(X_i - \hat{X})(Y_i - \hat{Y})}{\sum_{i}^{N}(X_i - \hat{X})^{2}}
$$

Therefore, the slope of the linear model $\hat{\beta_1}$ satisfies:

$$
\hat{\beta_1} = \frac{s_{X,Y}^{2}}{s_{X}^{2}} = r\frac{s_{Y}}{s_{X}}
$$

which shows that the slope of the linear model is governed by the correlation coefficient $r$. We can use the obtained model to create a predictor for $Y$:

$$
\hat{Y}(X) = \hat{\beta_1}X + \hat{\beta_0}
$$

<left><img src="Images\EX1511.png" alt="Alt Text" width="650" height="350"></left>

<left><img src="Images\EX1512.png" alt="Alt Text" width="650" height="80"></left>

<left><img src="Images\EX1513.png" alt="Alt Text" width="650" height="400"></left>

<left><img src="Images\EX1514.png" alt="Alt Text" width="650" height="80"></left>

**Goodness of fit**

The least squared regression problem has a goodness of fit metric called the coefficient of determination, $R^{2}$. As we have just performed least squares regression on the Hubble data, it would be a good idea to find the value of $R^{2}$ for our solution.

The coefficient of determination is defined as:

$$
R^{2} = 1 - \frac{SumSq_{\text {res}}}{SumSq_{\text {tot}}}
$$

$$
SumSq_{\text {res}} = \sum _ i^ N \left( Y_ i - \hat{Y}(X_ i) \right)^2
$$

is the sum of the squared residuals, and

$$
SumSq_{\text {tot}} = \sum _ i^ N \left( Y_ i - \bar{Y} \right)^2
$$

is the total sum of the squares (for $Y$). Can we find $R^{2}$ without performing another calculation? (Only one of these is true.)

<left><img src="Images\EX1515.png" alt="Alt Text" width="650" height="200"></left>

<left><img src="Images\EX1516.png" alt="Alt Text" width="650" height="400"></left>

**Final note:**

The value of $\hat{\beta_1}$ that you found is known as the Hubble constant. Since Hubble's first discovery of this relationship, astronomers have expended considerable effort to refine the value of this constant.

Today, we now know that the original constant that Hubble derived (and also the one that you derived, as you used the original data), is too large by a factor of between 6 and 7. Unknown to Hubble at the time, there are two types of Cepheid variable stars, and the difference between them needs to be accounted for in order to accurately estimate the distance to other galaxies.

Even today, the matter of Hubble's constant is not settled, as measurements of the constant based on nearby galaxies disagrees with measurements based on the cosmic microwave background (which tells us about the early universe, and thus much greater distances). Hubble's constant is critical to understanding the geometry of space-time in our universe, and so work continues to resolve this discrepancy!


If the true underlying relationship in your data is nonlinear, then attempting to fit a linear model may give poor results in the best circumstances. In the worst circumstances, it can be misleading. You may conclude that there is no correlation present in the data, or that the relationship is linear when in fact it is not.

Generally speaking, fitting a nonlinear relationship must be done using nonlinear regression. However, if the nonlinearity is simple, it can be possible to transform the nonlinear relationship into a linear one.

## 2.3 Correcting simple nonlinear relationships

If the true underlying relationship in your data is nonlinear, then attempting to fit a linear model may give poor results in the best circumstances. In the worst circumstances, it can be misleading. You may conclude that there is no correlation present in the data, or that the relationship is linear when in fact it is not.

Generally speaking, fitting a nonlinear relationship must be done using nonlinear regression. However, if the nonlinearity is simple, it can be possible to transform the nonlinear relationship into a linear one. Consider the following relationship:

$$
Y = \alpha e^{\beta X},
$$


where $\alpha$ and $\beta$ are two parameters. Attempting to directly fit this relationship with a linear model will preform terrible. But if we take the log on both sides, the original model will entail a linear form, i.e.,

$$
\ln {Y} = \beta X + \ln {\alpha },
$$

This looks linear, with $\beta$ being slope and $ln \alpha$ intercept. With this observation, we can transform the data points by letting $Y_ i' = \ln {Y_ i}$, for $i=1, \dots , N$. Under such a transformation, $Y_ i'$ and $X$ will have a linear relationship and we can employ linear regression on the problem. Let's consider another relationship:

$$
Y = \beta \ln {X} + \alpha ,
$$ 

in which if we transform $X$ to $X'$ using $X'_ i = ln{X_ i}$, the resulted model become a linear relationship with slope $\beta$ and intercept $\alpha$. The last example is to consider the following relationship:

$$
Y^{\gamma } = \alpha X^{\beta }.
$$

Then taking the log on both sides yields:

$$
\ln {Y} = \frac{\beta \ln {X} + \ln {\alpha }}{\gamma },
$$

which is a linear model with slope $\frac{\beta}{\gamma}$ and intercept $\frac{\ln{\alpha}}{\gamma}$.


**The effect of transformations on noise**

The above three cases cover a wide range of potential nonlinear relationships. If one or more of your variables are positively valued, then it can worthwhile transforming it to its logarithm in order to investigate a potential nonlinear relation in the data.

It should be noted that these transformations can change the statistics of any noise present in the data, and that noise can in turn interfere with the regression. For example, if we have the variates

$$
Y = \alpha e^{\beta X} \epsilon ,
$$

where $\epsilon$ is some multiplicative noise, then

$$
\ln {Y} = \beta X + \ln {\alpha } + \ln {\epsilon },
$$

and the noise is also transformed. If $\epsilon$ were, for example, log-normally distributed, then under the log transformation it would now be normally distributed. On the other hand, if we had:

$$
Y = \alpha \left( e^{\beta X} + \epsilon \right).
$$

Then:

$$
\ln {Y} = \ln {\left(e^{\beta X} + \epsilon \right)} + \ln {\alpha }.
$$

If $\epsilon$ is small, then this is still approximately linear, but if $\epsilon$ is large, we should not expect to see a linear relation. To handle such a situation correctly, we would need nonlinear regression.

For the following exercise, we will use the following data:

```python
Xs = np.array([ 0.387, 0.723, 1.00, 1.52, 5.20, 9.54, 19.2, 30.1, 39.5 ])

Ys = np.array([ 0.241, 0.615, 1.00, 1.88, 11.9, 29.5, 84.0, 165.0, 248 ])

N = 9
```

Each data point is one planet in our solar system (with the addition of the planetoid Pluto, which will be henceforth referred to as a planet for simplicity). The $X$ values are the semi-major axis of each planet's orbit around the Sun. A planetary orbit is elliptical in shape, and the semi-major axis is the longer of the two axes that define the ellipse. When the ellipse is nearly circular (which is true for most planets), the semi-major axis is approximately the radius of said circle. The $X$ values are measured in units of Astronomical Units (AU). One AU is very close to the average distance between the Sun and Earth (defined as 149597870700 meters), hence, the Earth's semi-major axis is essentially 1 AU due to its very circular orbit. The $Y$ values are the orbital period of the planet, measured in Earth years (365.25 days), so Earth also has a $Y = 1$ year.

<left><img src="Images\EX1517.png" alt="Alt Text" width="650" height="100"></left>

```python
# Creating Xs array
Xs = np.array([ 0.387, 0.723, 1.00, 1.52, 5.20, 9.54, 19.2, 30.1, 39.5 ])

# Creating Ys array
Ys = np.array([ 0.241, 0.615, 1.00, 1.88, 11.9, 29.5, 84.0, 165.0, 248 ])

# Storing the number of points
n = Xs.shape[0]

# Calculating Xs mean
Xs_mu = Xs.mean()

# Calculating Ys mean
Ys_mu = Ys.mean()

# Calculating Xs standard deviation
Xs_std = np.sqrt(np.sum((Xs-Xs.mean())**2)/(n-1))

# Calculating Ys standard deviation
Ys_std = np.sqrt(np.sum((Ys-Ys.mean())**2)/(n-1))

# Standard Xs
Xs_z = (Xs - Xs_mu)/Xs_std

# Standard Ys
Ys_z = (Ys - Ys_mu)/Ys_std

# Covariance between Xs and Ys
CovXY = np.sum((Xs-Xs.mean())*(Ys-Ys.mean()))/(n-1)

# Correlation between Xs and Ys
rXY = np.sum(Xs_z*Ys_z)/(n-1)

# Calculating beta_1
beta_1 = rXY*(Ys_std/Xs_std)

# Calculating beta_0
beta_0 = Ys_mu - beta_1*Xs_mu

# Calculating residuals
residuals = Ys - (beta_1*Xs + beta_0)

results = f"""
Xs mean: {Xs_mu} AU
Ys mean: {Ys_mu} year
Xs standard deviation: {Xs_std} AU
Ys standard deviation: {Ys_std} year
Covariance between Xs and Ys: {CovXY} AU year
Correlation between Xs and Ys: {rXY} unitless
Beta1: {beta_1}
Beta0: {beta_0}
"""

print(results)

:Xs mean: 11.907777777777776 AU
:Ys mean: 60.23733333333333 year
:Xs standard deviation: 14.494731601324823 AU
:Ys standard deviation: 89.54398623162808 year
:Covariance between Xs and Ys: 1283.4213123333334 AU year
:Correlation between Xs and Ys: 0.9888323018726735 unitless
:Beta1: 6.108701317117374
:Beta0: -12.503724461718768
```

```python
# Importing Seaborn
import seaborn as sns

# Creating plot
fig = sns.scatterplot(x=Xs, y=residuals)
 
# Adding labels to the axis and label to the plot
fig.set(xlabel ="X values", ylabel = "Residuals", title ='Residual plot');
```

<center><img src="Images\residualsX.png" alt="Alt Text" width="500" height="400"></center>

```python
# Importing Seaborn
import seaborn as sns

# Creating plot
fig = sns.scatterplot(x=Ys, y=residuals)
 
# Adding labels to the axis and label to the plot
fig.set(xlabel ="Y values", ylabel = "Residuals", title ='Residual plot');
```

<center><img src="Images\residualsY.png" alt="Alt Text" width="500" height="400"></center>


<left><img src="Images\EX1518.png" alt="Alt Text" width="650" height="180"></left>


```python
# Importing matplotlib pyplot
import matplotlib.pyplot as plt

# Importing statsmodels lib
import statsmodels.api as sm

# Creating a QQ-plot
sm.qqplot(Xs, line='s')
plt.title("X distribution")
plt.show()
```
Recall that the Q-Q plot compares the empirical distribution of the data to a theoretical normal distribution by comparing the empirical quantiles to the corresponding theoretical quantiles. If the empirical and theoretical distributions are the same, the data points on the plot will fall onto a $45^\circ$ line. In the above example, the **line='s'** option scales this line by the empirical mean and standard deviation.

<center><img src="Images\QQplot.png" alt="Alt Text" width="500" height="400"></center>

<left><img src="Images\EX1519.png" alt="Alt Text" width="650" height="150"></left>

<left><img src="Images\EX1520.png" alt="Alt Text" width="650" height="350"></left>

```python
# Creating Xs array and applying log
Xs = np.log(np.array([ 0.387, 0.723, 1.00, 1.52, 5.20, 9.54, 19.2, 30.1, 39.5 ]))

# Creating Ys array and applying log
Ys = np.log(np.array([ 0.241, 0.615, 1.00, 1.88, 11.9, 29.5, 84.0, 165.0, 248 ]))

# Storing the number of points
n = Xs.shape[0]

# Calculating Xs mean
Xs_mu = Xs.mean()

# Calculating Ys mean
Ys_mu = Ys.mean()

# Calculating Xs standard deviation
Xs_std = np.sqrt(np.sum((Xs-Xs.mean())**2)/(n-1))

# Calculating Ys standard deviation
Ys_std = np.sqrt(np.sum((Ys-Ys.mean())**2)/(n-1))

# Standard Xs
Xs_z = (Xs - Xs_mu)/Xs_std

# Standard Ys
Ys_z = (Ys - Ys_mu)/Ys_std

# Covariance between Xs and Ys
CovXY = np.sum((Xs-Xs.mean())*(Ys-Ys.mean()))/(n-1)

# Correlation between Xs and Ys
rXY = np.sum(Xs_z*Ys_z)/(n-1)

# Calculating beta_1
beta_1 = rXY*(Ys_std/Xs_std)

# Calculating beta_0
beta_0 = Ys_mu - beta_1*Xs_mu

# Calculating residuals
residuals = Ys - (beta_1*Xs + beta_0)

results = f"""
Xs mean: {Xs_mu} AU
Ys mean: {Ys_mu} year
Xs standard deviation: {Xs_std} AU
Ys standard deviation: {Ys_std} year
Covariance between Xs and Ys: {CovXY} AU year
Correlation between Xs and Ys: {rXY} unitless
Beta1: {beta_1}
Beta0: {beta_0}
"""

print(results)

:Xs mean: 1.4538802139806306 AU
:Ys mean: 2.181477789171877 year
:Xs standard deviation: 1.7248419253757807 AU
:Ys standard deviation: 2.586457153139843 year
:Covariance between Xs and Ys: 4.461228881643748 AU year
:Correlation between Xs and Ys: 0.9999998085102114 unitless
:Beta1: 1.4995325773381893
:Beta0: 0.0013370447605036695
```

<center><img src="Images\nonlinearXY.png" alt="Alt Text" width="500" height="400"></center>

We just found a linear relationship between $X'$ and $Y'$, but these are the transformed variables. What we would like to do now is find the nonlinear relationship between $X$ and $Y$ by transforming back.

Let:

$$
Y' = \kappa X' + \ln {\omega }.
$$

be your transformed model, with slope $\kappa$ and intercept $\ln {\omega }$.

<left><img src="Images\EX1521.png" alt="Alt Text" width="650" height="220"></left>

## 2.4 Multiple linear regression

If we have more than two observed variables, we can create a model that predicts one variable, $Y$, based on two or more other variables: $X_1, X_2, \ldots , X_ p$. Then, the model for one observation of the predicted variable, $Y_i$, can be written as:

$$
Y_ i = \beta _0 + \beta _1 X_{i,1} + \beta _2 X_{i,2} + \ldots + \beta _ p X_{i,p} +\epsilon _ i
$$

where $X_{i,j}$ is the j-th feature of data point $i$, and $p$ is the number of features. To streamline the derivation, we stack the data in the vector form. In particular, let:

$$
x_i = \begin{bmatrix}  1 \\ X_{i,1} \\ X_{i,2} \\ \vdots \\ X_{i,p} \end{bmatrix} \in \mathbb {R}^{p+1},
$$

$$
\beta = \begin{bmatrix}  \beta _0 \\ \beta _1 \\ \beta _2 \\ \vdots \\ \beta _ p \end{bmatrix} \in \mathbb {R}^{p+1}.
$$

Then the model can be written as

$$
Y_ i = \mathrm{{\boldsymbol x}}_ i^{\intercal }\mathrm{{\boldsymbol \beta }} + \epsilon _ i.
$$

Note that the leading 1 in $\mathrm{{\boldsymbol x}}_ i^{\intercal }$ multiplies by $\beta_0$ in $\beta$ to create the intercept. We could have kept the intercept out of the vector product as its own term, but this way reduces the number of terms.

Note that this equation is much like computing one element from the result of a matrix-vector product. We can use this to simplify the expression further, allowing us to write one equation for all observations. First, put all the  observations and noise terms  into their own vectors:

$$
y = \begin{bmatrix}  Y_1 \\ Y_2 \\ \vdots \\ Y_ N\end{bmatrix} \in \mathbb {R}^ N,
$$

$$
\epsilon = \begin{bmatrix}  \epsilon _1 \\ \epsilon _2 \\ \vdots \\ \epsilon _ N \end{bmatrix} \in \mathbb {R}^ N.
$$

Now, arrange each row vector  $\mathrm{{\boldsymbol x}}_ i^{\intercal }$ so that it forms one row of a larger matrix:

$$
\begin{bmatrix}  \rule[.5ex]{2.5ex}{0.5pt} &  \mathrm{{\boldsymbol x}}_1^{\intercal }&  \rule[.5ex]{2.5ex}{0.5pt} \\ \rule[.5ex]{2.5ex}{0.5pt} &  \mathrm{{\boldsymbol x}}_2^{\intercal }&  \rule[.5ex]{2.5ex}{0.5pt} \\ & \vdots & \\ \rule[.5ex]{2.5ex}{0.5pt} &  \mathrm{{\boldsymbol x}}_ N^{\intercal }&  \rule[.5ex]{2.5ex}{0.5pt} \end{bmatrix} \in \mathbb {R}^{N\times (p+1)}.
$$

Then the model can then be written as:

$$
\mathrm{{\boldsymbol y}} = {\boldsymbol X} \mathrm{{\boldsymbol \beta }} + \mathrm{{\boldsymbol \epsilon }}.
$$

**Exercises**

<left><img src="Images\EX1522.png" alt="Alt Text" width="650" height="520"></left>

<left><img src="Images\EX1523.png" alt="Alt Text" width="650" height="200"></left>

<left><img src="Images\EX1524.png" alt="Alt Text" width="650" height="400"></left>


**Exercise - Exoplanet mass data.**

For this exercise, we will perform multiple linear regression on some exoplanetary data to see if we can find a relationship that can predict the mass of an exoplanet.

```python
# Importing Libraries
import numpy as np

# Defining data
LogPlanetMass = np.array([-0.31471074,  1.01160091,  0.58778666,  0.46373402, -0.01005034,
         0.66577598, -1.30933332, -0.37106368, -0.40047757, -0.27443685,
         1.30833282, -0.46840491, -1.91054301,  0.16551444,  0.78845736,
        -2.43041846,  0.21511138,  2.29253476, -2.05330607, -0.43078292,
        -4.98204784, -0.48776035, -1.69298258, -0.08664781, -2.28278247,
         3.30431931, -3.27016912,  1.14644962, -3.10109279, -0.61248928])

LogPlanetRadius = np.array([ 0.32497786,  0.34712953,  0.14842001,  0.45742485,  0.1889661 ,
         0.06952606,  0.07696104,  0.3220835 ,  0.42918163, -0.05762911,
         0.40546511,  0.19227189, -0.16251893,  0.45107562,  0.3825376 ,
        -0.82098055,  0.10436002,  0.0295588 , -1.17921515,  0.55961579,
        -2.49253568,  0.11243543, -0.72037861,  0.36464311, -0.46203546,
         0.13976194, -2.70306266,  0.12221763, -2.41374014,  0.35627486])

LogPlanetOrbit = np.array([-2.63108916, -3.89026151, -3.13752628, -2.99633245, -3.12356565,
        -2.33924908, -2.8507665 , -3.04765735, -2.84043939, -3.19004544,
        -3.14655516, -3.13729584, -3.09887303, -3.09004295, -3.16296819,
        -2.3227878 , -3.77661837, -2.52572864, -4.13641734, -3.05018846,
        -2.40141145, -3.14795149, -0.40361682, -3.2148838 , -2.74575207,
        -3.70014265, -1.98923527, -3.35440922, -1.96897409, -2.99773428])

StarMetallicity = np.array([ 0.11 , -0.002, -0.4  ,  0.01 ,  0.15 ,  0.22 , -0.01 ,  0.02 ,
        -0.06 , -0.127,  0.   ,  0.12 ,  0.27 ,  0.09 , -0.077,  0.3  ,
         0.14 , -0.07 ,  0.19 , -0.02 ,  0.12 ,  0.251,  0.07 ,  0.16 ,
         0.19 ,  0.052, -0.32 ,  0.258,  0.02 , -0.17 ])

LogStarMass = np.array([ 0.27002714,  0.19144646, -0.16369609,  0.44468582,  0.19227189,
         0.01291623,  0.0861777 ,  0.1380213 ,  0.49469624, -0.43850496,
         0.54232429,  0.02469261,  0.07325046,  0.42133846,  0.2592826 ,
        -0.09431068, -0.24846136, -0.12783337, -0.07364654,  0.26159474,
         0.07603469, -0.07796154,  0.09440068,  0.07510747,  0.17395331,
         0.28893129, -0.21940057,  0.02566775, -0.09211529,  0.16551444])

LogStarAge = np.array([ 1.58103844,  1.06471074,  2.39789527,  0.72754861,  0.55675456,
         1.91692261,  1.64865863,  1.38629436,  0.77472717,  1.36097655,
         0.        ,  1.80828877,  1.7837273 ,  0.64185389,  0.69813472,
         2.39789527, -0.35667494,  1.79175947,  1.90210753,  1.39624469,
         1.84054963,  2.19722458,  1.89761986,  1.84054963,  0.74193734,
         0.55961579,  1.79175947,  0.91629073,  2.17475172,  1.36097655])

N = 30
```

**Choice of variable transformation.**

All of these observed quantities have been transformed by taking the natural logarithm. When performing linear regression, it can help to have a general idea on how the predictors contribute to the predicted quantity.

For example, if one were attempting to predict the sales of a store based on the population of surrounding region, then we might expect that the sales will be cumulative in the population variables. In this case, it would be best to leave these variables as they are, performing the linear regression directly on them.

However, in astronomy and physics, it is very common for the predicted variable to be multiplicative in the predictors. For example, the power that a solar cell produces is the product of the amount of solar radiation and the efficiency of the cell. In that case, it is better to transform the variables by taking the logarithm as discussed previously.

**LogPlanetMass** is the logarithm of the observed exoplanet's mass in units of Jupiter's mass. A LogPlanetMass of zero is an exoplanet with the same mass as Jupiter. Jupiter is used as a convenient comparison, as large gas giants are the most easily detected, and thus most commonly observed, kind of exoplanet. 

**LogPlanetRadius** is the logarithm of the observed exoplanet's radius in units of Jupiter's radius, for much the same reason.

**LogPlanetOrbit** is the logarithm of the observed planet's semi-major axis of orbit, in units of AU. 

**StarMetallicity** is the relative amount of metals observed in the parent star. It is equal to the logarithm of the ratio of the observed abundance of metal to the observed abundance of metal in the Sun. The Sun is a quite average star, so it serves as a good reference point. The most common metal to measure is Iron, but astronomers define any element that isn't Hydrogen or Helium as a metal. 

**LogStarMass** is the logarithm of the parent star's mass in units of the Sun's mass. 

**LogStarAge** is the logarithm of the parent star's age in giga-years.

```python
# Creating X vector
X = np.concatenate([
    np.ones(shape=(N,1)), 
    LogPlanetRadius.reshape(N,1),
    LogPlanetOrbit.reshape(N,1),
    StarMetallicity.reshape(N,1),
    LogStarMass.reshape(N,1),
    LogStarAge.reshape(N,1)
], axis=1)

# Creating y vector
y = LogPlanetMass.reshape(N,1)

# Calculating Beta
beta = np.linalg.inv(X.T@X)@X.T@y

# Printing Beta
print(beta[:,0])

: [ 0.15379303  1.40214538 -0.14099818 -1.59951364 -0.95610919 -0.46176234]
```
<left><img src="Images\EX1525.png" alt="Alt Text" width="650" height="300"></left>

## 2.5 Model selection and regularization

We consider a dataset of $p$ features and $N$ data points, denoted by $\{ (X_1,Y_1), \dots , (X_ N, Y_ N)\}$, where $X_i\in \mathbb R^p$, and $Y_ i\in \mathbb R$. To include the intercept in the model, we assume the first element of $X_i$ is 1, for all $i=1, \dots , N$. Then the linear regression model for this dataset can be written in a matrix form

$$
\mathrm{{\boldsymbol y}} = {\boldsymbol X} \mathrm{{\boldsymbol \beta }} + \mathrm{{\boldsymbol \epsilon }},
$$
 		 	 
where $\mathrm{{\boldsymbol \beta }} \in \mathbb R^{p}$, $\mathrm{{\boldsymbol \epsilon }} \in \mathbb R^{N}$, $\mathrm{{\boldsymbol y}}=(Y_1, \dots , Y_ N)^{\intercal }\in \mathbb R^{N}$, and data matrix $X$ is defined by ${\boldsymbol X}=(X_1, X_2, \dots , X_ N)^ T \in \mathbb R^{N\times p}$.

Furthermore, we assume that the elements of $\epsilon$ are i.i.d. with normal distribution $\mathrm{{\boldsymbol \epsilon }}_ i \sim \mathcal{N}(0,\sigma )$, for $i=1,\dots , N$, and that $\epsilon$ is independent of data $X$. When data matrix $X$ is full column rank, the least squares estimate of coefficient $\beta$ is given by:

$$
\hat{\mathrm{{\boldsymbol \beta }}} = ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1} {\boldsymbol X}^{\intercal }\mathrm{{\boldsymbol y}}.
$$
 		 	 
Then we claim that such an estimator $\hat{\mathrm{{\boldsymbol \beta }}}$ conditions on $X$ satisfies:

$$
\mathbb{E} [\hat{\beta} | x] =  \mathrm{{\boldsymbol \beta }},
$$

$$
Cov[\hat{\beta} | x] = \sigma ^2 ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1},
$$
 			 	 
which implies estimator $\hat{\mathrm{{\boldsymbol \beta }}}$ is conditionally unbiased.

<center><img src="Images\Proof001.png" alt="Alt Text" width="650" height="600"></center>

So far we have already found the conditional expectation and covariance matrix for estimator $\hat{\mathrm{{\boldsymbol \beta }}}$, but in $Cov[\hat{\beta} | x]$, we are still unaware of $\sigma$. The good news is that we can show that there exists a conditional unbiased estimator for $\sigma^2$, given by:

$$
\hat{\sigma }^2 = \frac{\sum _ i^ N (Y_ i - \mathrm{{\boldsymbol x}}_ i^{\intercal }\hat{\mathrm{{\boldsymbol \beta }}})^2}{N-p}.
$$

Now, we are ready to use t-test to test the null hypothesis that $\beta_j = 0$. Recall the covariance matrix ${Cov}[\hat{\mathrm{{\boldsymbol \beta }}}|{\boldsymbol X}] = \sigma ^2 ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1}$. We denote the j-th diagonal element of $({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1}$ by $\Sigma_{j}^{2}$. Then, under the null hypothesis, we have:

$$
\frac{\hat{\beta }_ j - 0}{\sigma \Sigma _ j} \sim \mathcal{N}(0,1)
$$

where $\sigma \Sigma _ j$  is the conditional standard deviation of $\hat{\beta }_ j$. Now the convariance estimator:

$$
\sigma^2 = \frac{{| | \mathrm{{\boldsymbol y}} - {\boldsymbol X} \hat{\mathrm{{\boldsymbol \beta }}} | |}^2}{N-p}
$$

is an estimator for $\sigma^2$, and

$$
(N-p)\frac{\hat{\sigma }^2}{\sigma ^2} \sim \chi ^2_{N-p}
$$

where $p$ is the number of columns in $\hat{X}$. If $Z \sim \mathcal{N}(0,1)$ and $\omega \sim \chi^2_ n$ then:

$$
\frac{Z}{\sqrt{\frac{\omega }{n}}} \sim t_n
$$

is t-distributed with $n$ degrees of freedom. Therefore:

$$
T_j = \frac{ \frac{\hat{\beta }_ j - 0}{\sigma \Sigma _ j} }{ \sqrt{\frac{(N-p)\frac{\hat{\sigma }^2}{\sigma ^2}}{N-p}} } = \frac{\hat{\beta }_ j}{\hat{\sigma } \Sigma _ j}
$$

is $t$ distributed with $N - p$ degrees of freedom and can be used as a t-test to test the hypothesis that $\beta_j = 0$.

<left><img src="Images\EX1526.png" alt="Alt Text" width="650" height="350"></left>

<left><img src="Images\EX1527.png" alt="Alt Text" width="650" height="250"></left>

<left><img src="Images\EX1528.png" alt="Alt Text" width="650" height="250"></left>

```python
# Calculating sigma hat
sigma_hat = np.sqrt(np.sum(((y - X@beta)**2)/(N-X.shape[1])))

# Calculating Sigma vector
Sigma_vec = np.sqrt(np.linalg.inv(X.T@X).diagonal()).reshape(X.shape[1], 1)

# Calculating T-statistics
T_vec = beta/(sigma_hat*Sigma_vec)

# Features
features_name = ['intercept', 'LogPlanetRadius', 'LogPlanetOrbit', 'StarMetallicity', 'LogStarMass', 'LogStarAge']

# Printing Beta
print([f"{feature}: {round(value,3)}" for feature, value in zip(features_name, T_vec[:,0])])

: ['intercept: 0.111', 'LogPlanetRadius: 4.895', 'LogPlanetOrbit: -0.392', 'StarMetallicity: -1.279', 'LogStarMass: -0.856', 'LogStarAge: -1.243']
```

## 2.6 Convexity

Recall that for the Ordinary Least Squares (OLS) problem:

$$
\hat{w} = \arg \min _{\mathrm{{\boldsymbol w}}} \sum _{i=1}^ N \left(Y_ i - \mathrm{{\boldsymbol x}}_ i^{\intercal }\mathrm{{\boldsymbol w}}\right)^2,
$$

we can find the minimum by writing the sum of the squares, $f(w)$, as:

$$
f(w) = \sum _{i=1}^ N \left(Y_ i - \mathrm{{\boldsymbol x}}_ i^{\intercal }\mathrm{{\boldsymbol w}}\right)^2.
$$

Then, we take the derivative of the sum-of-squares and set it to zero:

$$
\left.\frac{\partial f}{\partial \mathrm{{\boldsymbol w}}}\right|_{\mathrm{{\boldsymbol w}} = \hat{\mathrm{{\boldsymbol w}}}} = 0
$$

Note: This $f$ quantity is our optimization target, as we're trying to make it as small as possible. In the parlance of the field of optimization, this $f$ is often called the “loss function" (sometimes “cost function" is used). The idea is that it quantifies something that is “lost" or “cost" by a choice of  and we're trying to minimize this loss.

This is a powerful method of solving the problem, as we can find the solution as an equation, and then simply evaluate the equation. Can we use this technique for all optimization problems? And if not, what can we use it with OLS?

Rather than answer the question about OLS immediately, we will first investigate a property of functions called convexity. It turns out that this property is exactly what we're looking for. If the loss function is convex, then we can set the gradient to zero in order to find the minimum.

<left><img src="Images\EX1529.png" alt="Alt Text" width="650" height="450"></left>

<left><img src="Images\EX1530.png" alt="Alt Text" width="650" height="350"></left>

<left><img src="Images\EX1531.png" alt="Alt Text" width="650" height="350"></left>

<left><img src="Images\EX1533.png" alt="Alt Text" width="650" height="500"></left>

<left><img src="Images\EX1532.png" alt="Alt Text" width="650" height="600"></left>

<left><img src="Images\EX1534.png" alt="Alt Text" width="650" height="500"></left>

<left><img src="Images\EX1535.png" alt="Alt Text" width="650" height="300"></left>

We seek a condition that implies that any critical point is unique, and that this critical point is a minimum (and thus the global minimum). This condition is called convexity. If we arrange our optimization problems such that they are convex, then they will be much easier to solve. Note that a convex function may not have any critical points at all, convexity only guarantees that the number of critical points is either zero or one.

<center><img src="Images\Curves.png" alt="Alt Text" width="650" height="600"></center>

We see that the two minima, A and B, correspond to points where the derivative curve intersects the w-axis. Observe, also, that the point where the derivative curve intersects the w-axis, always corresponds to a positive slope, showing that $f^{\prime \prime }(w) > 0$ there.

This is important: at the global minimum the slope of the derivative curve is positive, so the only way for there to be another (local) minimum is if the curve dips back below the -axis so that it can again intersect the -axis with positive slope. And, in fact, we see this in the curve above. After point A, the curve turns over, goes below the axis, and then turns back and intersects the axis at point B with positive slope.

The only way for the derivative curve to do this is for it to have, at some point, a negative slope. There means that for a second minimum to exist somewhere, we must have that $f^{\prime \prime }(w) < 0$ for some range of $w$ values.

This allows us to create the following sufficient condition for the uniqueness of the minimum of a loss function: If the second derivative is non-negative everywhere ($f^{\prime \prime }(w) \geq 0$ for all $w$) then this implies that the critical point of the function (if it exists) is the unique global minimum.

Such a loss-function is called “convex". For a twice differentiable loss function $f$, it is convex if and only if the second derivative is non-negative everywhere. We should immediately note that this is not the only way to define convexity.

<left><img src="Images\Text001.png" alt="Alt Text" width="650" height="600"></left>

<left><img src="Images\Text002.png" alt="Alt Text" width="650" height="500"></left>

<left><img src="Images\EX1536.png" alt="Alt Text" width="650" height="400"></left>

<left><img src="Images\Text003.png" alt="Alt Text" width="650" height="600"></left>

<left><img src="Images\Text004.png" alt="Alt Text" width="650" height="600"></left>

<left><img src="Images\EX1537.png" alt="Alt Text" width="650" height="450"></left>

<left><img src="Images\EX1538.png" alt="Alt Text" width="650" height="250"></left>

Solution:

Function (A) is indeed convex, as the second derivative of $e^{x}$ is $e^{x}$ which is positive everywhere. However, it does not have a global minimum, as it has no critical points: $e^ x \neq 0$ everywhere.

Function (B) is also convex. Remember that the definitions of convexity use greater-than-or-equal relations. The second derivative of a line may be zero, but this is technically non-negative as so the line is convex. There is a more restrictive form of convexity called strict convexity that the line does not posses. Much like the exponential, the line has no global minimum despite being convex, as the derivative is non-zero everywhere and thus there are no critical points.

Function (C) is a peicewise function, that is composed of two quadratics. Each quadratic, on its own, is a convex function with a global minimum. However, the peicewise combination is not convex. We can't use the second derivate criterion as the peicewise function is not differentiable at the break. We can use the chord or tangent lower bound criteria, and this function fails both. Despite this, the function has a global minimum.

Function (D) is not convex, though it may look like it at first. Careful examination of the curve will show that it fails the chord and tangent lower bound criteria (consider the tangent near the edges of the bell). If we take the second derivative, we also find that there are places where it is negative, so it fails this criterion too. However, it does have a global minimum, as there is just one critical point and the second derivative is positive at that point.

Something important to consider is that if we take function (D), and transform it using a logarithm, $f(w)' = -\ln {(-f(w))} = x^2$ then we get a convex  function with a global minimum. It is common to use transformation to turn non-convex problems into convex one, and the logarithm is a very common transformation for doing this.

## 2.7. Multidimensional convexity and local optimization

**Multidimensional Taylor expansion**

Now let us consider loss functions that are parameterized by multiple weights. We'll arrange the weights such that they form a column vector, $\mathrm{{\boldsymbol w}} \in \mathbb R^ d$. Then, the loss function is a scalar-valued function of this vector of parameters: $f(\mathrm{{\boldsymbol w}})\in \mathbb R$. We can expand $f(w)$ around a point in parameter space, $w_0$, through use of the multidimensional Taylor expansion:

<center><img src="Images\EX1539.png" alt="Alt Text" width="650" height="250"></center>

**Critical points**

Much of the previous discussion transfers over to the multidimensional case. To start, the critical points of $f(w)$ are defined as the solutions, $\mathrm{{\boldsymbol w}}'$, to $\displaystyle \displaystyle (\nabla f)(\mathrm{{\boldsymbol w}}') = 0$, where the value on the right hand side is the zero vector. At a critical point, the Taylor expansion is:

<center><img src="Images\EX1540.png" alt="Alt Text" width="650" height="500"></center>

<center><img src="Images\EX1541.png" alt="Alt Text" width="650" height="350"></center>

**Saddle points**

In the one dimensional case we found that if the second derivative is zero at a critical point, then that critical point may be a saddle point. In multiple dimensions, we can get more information out of the Hessian matrix and are thus able to make stronger statements on the existence of a saddle point.

First, let us examine a relaxation on the positive definite property of a matrix. A matrix $H$ is called a positive semi-definite matrix if and only if $\displaystyle \displaystyle \mathrm{{\boldsymbol v}}^{\intercal }{\boldsymbol H} \mathrm{{\boldsymbol v}} \geq 0$ for all non-zero vectors $\mathrm{{\boldsymbol v}} \neq 0$. In addition, a real-valued symmetric matrix is positive semi-definite if and only if all the eigenvalues of the matrix are non-negative.

Certainly all positive definite matrices are also positive semi-definite. But if a real-valued symmetric matrix has all positive eigenvalues except for one or more zero eigenvalues, then the matrix is positive semi-definite but not positive definite. More importantly, if $\displaystyle \displaystyle \mathrm{{\boldsymbol v}}^{\intercal }{\boldsymbol H} \mathrm{{\boldsymbol v}} = 0$ for all vectors $\mathrm{{\boldsymbol v}}$, then the matrix $H$ is also positive semi-definite.

This leaves us in a situation much like we were in with the one dimensional case. A Hessian matrix this is entirely zero could be a minimum, or a maximum, or a saddle point. For example, the loss functions:

$$
f(w_1, w_2) = w_{1}^{4} + w_{2}^{4}
$$

$$
f(w_1, w_2) = w_{1}^{4} - w_{2}^{4}
$$

$$
f(w_1, w_2) = - w_{1}^{4} - w_{2}^{4}
$$

all have zero matrices for their Hessian matrix at the critical point $w_1=w_2=0$; but the first has a minimum, the second a saddle point, and the third a maximum.

While a zero matrix fits the definition of a positive semi-definite matrix, all the eigenvalues of a zero matrix must be zero. This suggests that if we examine the eigenvalues directly, we might get a clearer picture.

Recall that each eigenvalue is associated with an eigenvector, and the eigenvectors describe a direction that the matrix acts upon. For the Hessian matrix at a critical point, we can interpret its eigenvalue-eigenvector pair as follows: the eigenvector is a vector that points towards a direction away from the critical point, and the eigenvalue shows if the curvature of $f(w)$ is positive, negative, or zero in that direction.

Note: We will use the word “curvature" here to mean the rate of change of the gradient (so the sign and magnitude of the second derivative). In geometry, there are many different notions of curvature, so you may see the term defined differently in other contexts.

If a Hessian matrix has at least one positive eigenvalue, then we know that there is a direction away from the critical point where the loss function curves upwards. Meanwhile, if the same Hessian matrix also has at least one negative eigenvalue, then we know that there is a direction away from the critical point where the loss function curves downwards. A mixture of curving upwards and downwards is the definition of a saddle point, so we now know that the critical point associated with this Hessian is a saddle point.

Take, for example, the following loss function, shown below:

<center><img src="Images\HighDimensionCurve.png" alt="Alt Text" width="650" height="400"></center>

In this example, a saddle point is located at the origin (where the black vertical line intersects the surface). The black vector is an eigenvector that points in the direction of positive curvature. The red vector is an eigenvector that points in the direction of negative curvature.

In general, a real-valued symmetric matrix with both positive and negative eigenvalues is called an indefinite matrix, and the product $\mathrm{{\boldsymbol v}}^{\intercal }{\boldsymbol H} \mathrm{{\boldsymbol v}}$ for any specific $\mathrm{{\boldsymbol v}}$ may be positive, negative, or zero.

Can we determine if a real-valued symmetric matrix (of any general size) is indefinite or not using only the determinant of the matrix?

<center><img src="Images\EX1542.png" alt="Alt Text" width="650" height="100"></center>

We can now find one condition for convexity in multiple dimensions: $f$ is convex if and only if the Hessian is positive semi-definite everywhere. Just as in the one-dimensional case, we can write the Taylor expansion with a remainder term, $R_1(\mathrm{{\boldsymbol w}};\mathrm{{\boldsymbol w}}_0)$:

$$
f(w) = f(\mathrm{{\boldsymbol w}}_0) + (\nabla f)(\mathrm{{\boldsymbol w}}_0) (\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}_0) + R_1(\mathrm{{\boldsymbol w}};\mathrm{{\boldsymbol w}}_0).
$$

It turns out that there is some $\mathrm{{\boldsymbol w}}_{*}$ that is an element of the line between $w$ and $w_0$ such that

$$
R_1(\mathrm{{\boldsymbol w}};\mathrm{{\boldsymbol w}}_0) = \frac{1}{2} (\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}_0)^{\intercal }(\nabla \nabla f)(\mathrm{{\boldsymbol w}}_{*}) (\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}_0).
$$

If the loss function is convex, then the Hessian is positive semi-definite, so by definition $R_1(\mathrm{{\boldsymbol w}};\mathrm{{\boldsymbol w}}_0) \geq 0$; and, in analogy to the one dimensional case, we find a lower bound

$$
f(w) \geq f(\mathrm{{\boldsymbol w}}_0) + (\nabla f)(\mathrm{{\boldsymbol w}}_0) (\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}_0).
$$
 			 	 
In two dimensions, the expression on the right is an equation of a plane. In more than two dimensions, it is the equation of a hyperplane. Thus, we have found a tangent hyperplane that is a lower bound for the loss function. And this tangent hyperplane exists for all choices of $w$. Much like in the one dimensional case, we have the following condition: the loss function $f(w)$ is convex if and only if there exists a lower bound tangent hyperplane for all $w$.

**Hessian example**

<center><img src="Images\EX1545.png" alt="Alt Text" width="650" height="400"></center>

## 2.8. Quadratic minimization and gradient descent

Although convexity ensures that a loss function has some nice properties – such as a unique (global) minimum – it does not guarantee that the loss function can be solved exactly. For example, the following loss function:

$$
f(x) = x^4 - \frac{x^3}{5} + \frac{x^2}{4} - \frac{x}{3}
$$

is convex; but, as a forth order polynomial, it has no simple closed form equation for the minimum.

Instead of trying to find a single equation for the minimum, we can make a guess at the minimum and iteratively refine this guess to bring it closer to the true minimum. Let the initial guess for the minimum be denoted as . We can now approximate the loss function and use this approximation to refine our guess. Previously, we found that a convex loss function has a lower bound:

$$
f(w) \geq f(w_0) + f'(w_0) (w - w_0);
$$

however, this would not make a particularly useful approximation for our current problem, as the line equation on the right hand side has no minimum. Instead, we can go one order up. Recall that the lower bound was derived from the Taylor expansion:

$$
f(w) = f(w_0) + f'(w_0) (w - w_0) + \frac{1}{2} f^{\prime \prime }(w_0) (w - w_0)^2 + \mathcal{O}(|w-w_0|^3).
$$

We can create an approximation, denoted by $g_0$, by truncating this expansion at the second order:

$$
g_0(w) = f(w_0) + f'(w_0) (w - w_0) + \frac{1}{2} f^{\prime \prime }(w_0) (w - w_0)^2
$$

As $f$ is convex, we know that $f^{\prime \prime }(w_0) \geq 0$, and so $g_0$ will also be convex, and has a global minimum that we can find.

**Newton's method of minimization**

<center><img src="Images\EX1546.png" alt="Alt Text" width="650" height="700"></center>

The iterative procedure of Newton's method requires some condition to stop the iteration. There is no single correct condition, but a common condition is to stop when the norm of the gradient is below some threshold, $\epsilon$, that is very close to zero:

$$
{| | (\nabla f)(\mathrm{{\boldsymbol w}}_ t) | |}^2 < \epsilon
$$

Another is to also compute the value of the loss function, $f(w_t)$, at each iteration, and then stop when the difference between the functions is below a threshold, $\epsilon$, that is close to zero:

$$
f(\mathrm{{\boldsymbol w}}_{t-1}) - f(\mathrm{{\boldsymbol w}}_{t}) < \epsilon
$$

**Gradient descent**

<center><img src="Images\EX1547.png" alt="Alt Text" width="650" height="550"></center>

**Non-convex functions**

<center><img src="Images\EX1548.png" alt="Alt Text" width="650" height="250"></center>


## 3. Module 2: Genomics and High-Dimensional Data and Visualization of High-Dimensional Data

## 3.1. Introduction

Dimension reduction refers to a set of techniques which can transform high-dimensional data into their representative low-dimensional data. During the process, <mark>some information of the original data is discarded but some main characteristics of the original data is preserved</mark>.

Dimension reduction is important because processing and analyzing high-dimensional data can be computationally intractable. Dimension reduction is very useful in dealing with a large number of observations and variables, hence it is widely used in many fields such as signal processing, machine learning, and bioinformatics.

Three dimension reduction techniques will be introduced:

- Principal Component Analysis (PCA);
  - Tries to project the original high-dimensional data into lower dimensions by capturing the most prominent variance in the data.
- Multidimensional Scaling (MDS);
  - Technique for reducing data dimensions while attempting to preserve the relative distance between high-dimensional data points
- Stochastic Neighbor Embedding (SNE).
  - Non-linear technique to “cluster" data points by trying to keep similar data points close to each other.

PCA and classical MDS share similar computations: they both use the spectral decomposition of symmetric matrices, but on different input matrices. You will think through some of the mathematics in this lecture. For demonstration of implementations, see the recitation at the end of this module.

## 3.2 Expectation and Covariance of a Random Vector

<center><img src="Images\EX2000.png" alt="Alt Text" width="650" height="350"></center>
<center><img src="Images\EX2001.png" alt="Alt Text" width="650" height="450"></center>
<center><img src="Images\EX2002.png" alt="Alt Text" width="650" height="450"></center>
<center><img src="Images\EX2004.png" alt="Alt Text" width="650" height="400"></center>
<center><img src="Images\EX2005.png" alt="Alt Text" width="650" height="400"></center>
<center><img src="Images\EX2006.png" alt="Alt Text" width="650" height="400"></center>
<center><img src="Images\EX2007.png" alt="Alt Text" width="650" height="350"></center>
<center><img src="Images\EX2008.png" alt="Alt Text" width="650" height="350"></center>
<center><img src="Images\EX2009.png" alt="Alt Text" width="650" height="350"></center>
<center><img src="Images\EX2010.png" alt="Alt Text" width="450" height="350"></center>
<center><img src="Images\EX2011.png" alt="Alt Text" width="650" height="450"></center>
<center><img src="Images\EX2012.png" alt="Alt Text" width="650" height="600"></center>
<center><img src="Images\EX2013.png" alt="Alt Text" width="650" height="600"></center>
<center><img src="Images\EX2014.png" alt="Alt Text" width="650" height="600"></center>
<center><img src="Images\EX2015.png" alt="Alt Text" width="650" height="600"></center>
<center><img src="Images\EX2016.png" alt="Alt Text" width="650" height="600"></center>
<center><img src="Images\EX2017.png" alt="Alt Text" width="650" height="700"></center>
<center><img src="Images\EX2018.png" alt="Alt Text" width="650" height="700"></center>
<center><img src="Images\EX2019.png" alt="Alt Text" width="650" height="300"></center>

## 3.3 Principal component analysis (PCA)

Principal component analysis (PCA) is often used to find a low dimensional representation of data that maximizes the spread of the projected data.

The first principal component (PC1) is the direction of the largest variance of the data. The second principal component (PC2) is perpendicular to the first principal component and is the direction of the largest variance of the data among all directions that are perpendicular to the first principal component. The third principal component (PC3) is perpendicular to both first and second principal components and is in the direction of the largest variance among all directions that are perpendicular to both the first and second principal components. This can continue until we obtain as many principal components as the dimension of the original space in which the data is given, i.e. an orthogonal basis of the data space consisting of principal components. However, we are generally interested in the fewer dimensions as the original space, for example, 2 or 3 dimensions for visualization, or few than 100 dimensions out of thousands of dimensions of the original space, as you will see in the recitation and the analysis at the end of this module.

Principal component analysis can be formulated in the following three equivalent ways. For simpliciy, we will only formulate the problem for the first principal component (PC1).

Let $\mathbf{x}^{(1)},\mathbf{x}^{(2)}, \dots , \mathbf{x}^{(n)}\in \mathbb {R}^ p$ denote the $n$ data points in $p$ dimensional space. Without loss of generality, we assume these data is centered at the origin, i.e., $\sum _{i=1}^ n \mathbf{x}^{(i)} = 0$. The first principal component is the line spanned by a unit vector $w \in \mathbb {R}^ p$ such that:

- $w$ minimizes the sum of squared residuals of the orthogonal projections of data $\mathbf{x}^{(1)}$ onto $w$:

$$
\min _{\mathbf{w}\in \mathbb {R}^ p} \sum _{i=1}^ n \left\|  \mathbf{x}^{(i)}-(\mathbf{x}^{(i)}\cdot \mathbf{w})\mathbf{w} \right\| ^2
$$

- $w$ maximizes the sum of squared norms of the orthogonal projections of data $\mathbf{x}^{(1)}$ onto $w$:
  
$$
\max _{\mathbf{w}\in \mathbb {R}^ p} \sum _{i=1}^ n \left\|  (\mathbf{x}^{(i)}\cdot \mathbf{w}) \right\| ^2
$$

<center><img src="Images\EX2020.png" alt="Alt Text" width="650" height="400"></center>

- $w$ is an eigenvector corresponding to the largest eigenvalue of the the sample covariance matrix $\mathbf{S}$:

$$
\mathbf{S} = \frac{1}{n-1}\mathbb {X}^ T \mathbb {X}\qquad \text {where } \mathbb {X}\, =\,  \begin{pmatrix}  \leftarrow & (\mathbf{x}^{(1)})^ T& \rightarrow \\ \leftarrow & (\mathbf{x}^{(2)})^ T& \rightarrow \\ & \vdots & \\ \leftarrow & (\mathbf{x}^{(n)})^ T& \rightarrow \\ \end{pmatrix}.
$$

**Remark**: Note that the sample covariance matrix has different definitions (dividing by $n$ or $n-1$), but the eigenvectors and the order of the eigenvalues are not affected by this overall rescaling. Hence, if our goal is only to find the principal components, we are free to choose any scalar multiple of sample covariance matrix, e.g. $\mathbb {X}^ T \mathbb {X}$.

<center><img src="Images\EX2021.png" alt="Alt Text" width="650" height="550"></center>

To find the second principal component, we can repeat the process on the data projected into the space that is orthogonal to $PC1$, and so on. Or equivalently, the goal of PCA is to find an orthnormal basis $\{ \mathbf{w}^1, \mathbf{w}^2,\ldots , \mathbf{w}^ k\}$ of a k-dimensional subspace such that the squared residuals of the projection of the data into this subspace is minimized, or such that the squared norm of the projection of the data into this subspace is maximized, or such that this $k$ orthonormal vectors are eigenvectors corresponding to the $k$ largest eigenvalues of the sample covariance matrix. Notice that by definition, PCA gives linear projections into lower dimensional subspaces.

**Remark**: Finally, PCA applies even when for data that do not come with any labels.

<center><img src="Images\EX2022.png" alt="Alt Text" width="650" height="450"></center>
<center><img src="Images\EX2023.png" alt="Alt Text" width="650" height="450"></center>
<center><img src="Images\EX2024.png" alt="Alt Text" width="650" height="550"></center>
<center><img src="Images\EX2025.png" alt="Alt Text" width="650" height="550"></center>

## 3.4. Multidimensional Scaling (MDS)
  
Multidimensional scaling (MDS) is a non-linear dimensionality reduction method to extract a lower-dimensional configuration from the measurement of pairwise distances (dissimilarities) between the points in a dataset.

**Problem formulation**

For a dataset of points $\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots , \mathbf{x}^{(n)}$, with $\mathbf{x}^{(i)}\in \mathbb R^ p$, the distance matrix  consists of the elements of distances between each pair of the points, i.e., $d_{ij}= \| \mathbf{x}^{(i)}-\mathbf{x}^{(j)}\|$.

The objective of MDS is to find points $\mathbf{y}^{(1)}, \dots , \mathbf{y}^{(n)} \in \mathbb {R}^ q$ in a lower-dimensional space $(q < p)$, such that the sum of all pairwise distances:

$$
\sum _{i=1}^ n\sum _{j=1}^ n(d_{ij}-\lVert \mathbf{y}^{(i)} - \mathbf{y}^{(j)} \rVert _2)^2
$$

is minimized. The resulting points $\mathbf{y}^{(1)}, \dots , \mathbf{y}^{(n)}$ are called a **lower-dimensional embedding** of the original data points in space $\mathbb {R}^ q$. In general, the dimension $q = 1, 2, 3$, in such a way the resulting representation can be visualized on a scatter plot.

Note that when the distance matrix **D** is given, the MDS technique can operate without the original dataset $\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots , \mathbf{x}^{(n)}$.

**Variation of objectives**

Besides the above classical MDS, there are variations of MDS which use different objective functions.

- **Weighted MDS** uses the objective function

$$
\sum _{i=1}^ n\sum _{j=1}^ n w_{ij}(d_{ij}-\lVert \mathbf{y}^{(i)} - \mathbf{y}^{(j)} \rVert _2)^2,
$$
 
where $w_{ij} \geq 0$ is the assigned weight.

- **Non-metric MDS** uses the objective function

$$
\sum _{i=1}^ n\sum _{j=1}^ n (\theta (d_{ij})-\lVert \mathbf{y}^{(i)} - \mathbf{y}^{(j)} \rVert _2)^2,
$$
 
in which we also optimize the objective over an increasing function $\theta (\cdot )$.

In the both above non-classical MDS, the objective functions are non-convex, thus we will only consider the classical MDS in this lecture.

**Centered configuration and Gram matrix**

The solution to the classical MDS problem is not unique, because the solution is invariant under translations, rotations, and reflections. For example, if $\mathbf{Y}=(\mathbf{y}^{(1)}, \dots , \mathbf{y}^{(n)})^ T \in \mathbb R^{n\times q}$ is a solution, then for any scalar $c\in \mathbb R$, the configuration $\mathbf{Y}+ c E$ is also a solution, where matrix $E \in \mathbb R^{n\times q}$ consisting of all one elements.

Therefore, in the following we focus on the **centered configuration**, which is the configuration satisfying that for each $k$,

$$
\sum _{i=1}^ n \mathbf{x}^{(i)}_ k=0,
$$
 
with $\mathbf{x}^{(i)}_ k$ being the k-th component of point $\mathbf{x}^{(i)}$.

Then, for a given centered configuration $\mathbb {X}=(\mathbf{x}^{(1)}, \dots , \mathbf{x}^{(n)})^ T \in \mathbb R^{n\times p}$. We define the **Gram matrix** $B = \mathbb {X}\mathbb {X}^ T$. The element of the Gram matrix is the inner product between each pair of points in $\mathbb {X}$, i.e., the element of $B$ is:

$$
b_{ij} = \mathbf{x}^{(i)T}\mathbf{x}^{(j)}.
$$

We note that the Gram matrix is symmetric and positive semidefinite. Moreover, for a centered configuration $\mathbb {X}$, we can show that the corresponding distance matrix $D=(d_{ij})$ and the Gram matrix $B$ admit the relation:

$$
B = -\frac{1}{2} \left( I - \frac{1}{n} \mathbf{1} \mathbf{1}^ T\right) D^2 \left( I - \frac{1}{n} \mathbf{1} \mathbf{1}^ T\right),
$$
 		 	 
where $I$ is the identity matrix, vector $\mathbf{1}\in \mathbb R^{n}$ consisting of all one elements, and $D^{2}$ is the matrix with element $D^{2}_{ij} = (d_{ij})^2$.

## 3.5. Solving MDS

From the previous discussion, given the distance matrix $D$ of a dataset $\mathbb {X}$, we can find the Gram matrix of the corresponding centered configuration by:

$$
B=\mathbb {X}\mathbb {X}^ T =-\frac{1}{2}H D^2 H,
$$

where the double centering matrix $H= I - \frac{1}{n} \mathbf{1} \mathbf{1}^ T$, and $D^{2}$ is the matrix with element $D^{2}_{ij} = (d_{ij})^2$. Recall that the Gram matrix conveys the information of the length of each point and also the pairwise angles between all the points. Such information is therefore enough to uniquely determine the centered configuration, up to rotation and reflection.

In the MDS problem, given the dataset $\mathbb {X}\in \mathbb R^{n\times p}$, we want to find its lower-dimension embedding $\mathbf{Y}\in \mathbb R^{n\times q}$ $(q > p)$, such that the distance (or configuration encoded in $B$) are preserved as much as possible. This is equivalent to solve the problem:

$$
\min _{\mathbf{Y}\in \mathbb R^{n\times q}} \| B-\mathbf{Y}\mathbf{Y}^ T\| _ F^2,
$$

where $\| .\| _ F$ is the Frobenius norm. Note that as $\| B-\mathbf{Y}\mathbf{Y}^ T\| _ F^2= \text {trace}(B-\mathbf{Y}\mathbf{Y}^ T)^2$, thus another form of this problem is:

$$
\min _{\mathbf{Y}\in \mathbb R^{n\times q}} \text {trace}(B-\mathbf{Y}\mathbf{Y}^ T)^2,
$$

It is well known that the above optimization problems can be solved by the spectral decomposition. Specifically, the Gram matrix $B$ have eigenvalue decomposition $B=V\Lambda V^ T$, where $V$ consists of eigenvalues of $B$, and $\Lambda$ is a diagonal matrix containing eigenvalues of $B$. Note that the eigenvalues of $B$ are non-negative, since $B$ is a positive semidefinite matrix.

We choose $q$ largest eigenvalues in $\Lambda$ to construct the diagonal matrix $\Sigma _1 \in \mathbb R^{q\times q}$, and the corresponding eigenvectors to construct matrix $V_1=(v_1,\dots , v_ q)\in \mathbb R^{n\times q}$.

Then, the best rank $q$ approximation of $B$ is given by $\mathbf{Y}\mathbf{Y}^ T=V_1\Sigma _1V_1^ T$, and the solution to the forementioned optimization problem is:

$$
\mathbf{Y}=V_1\Sigma _1^{1/2},
$$
 
which is also the solution to the MDS.

<center><img src="Images\EX2026.png" alt="Alt Text" width="650" height="500"></center>
<center><img src="Images\EX2027.png" alt="Alt Text" width="650" height="350"></center>
<center><img src="Images\EX2028.png" alt="Alt Text" width="650" height="350"></center>

## 3.6. Stochastic Neighbor Embedding (SNE) and t-distributed Stochastic Neighbor Embedding (t-SNE)

**Stochastic neighbor embedding (SNE)** is a probabilistic approach to dimensional reduction that places data points in high dimensional space into low dimensional space while preserving the identity of neighbors. That is, SNE attempts to keep nearby data points nearby, and separated data points relatively far apart.

The idea of SNE is to define a probability distribution on pairs of data points in each of the original high dimensional space and the target low dimensional space, and then determine the placement of objects in low dimension by minimizing the “difference' of the two probability distributions.

More precisely:

- 1. Input: Distance matrix $D_{ij}$ of data in a p-dimensional space.
- 2. **In high dimension**: In the p-dimensional space, center a Gaussian distribution, $\exp (-\left\|  \mathbf{x}-\mathbf{x}^{(i)} \right\| ^2)$, on each data point $\mathbf{x}^{(i)}$. For each data point $\mathbf{x}^{(i)}$, define the probability of another data point $\mathbf{x}^{(j)}$ being its neighbor to be:

$$
p_{ij}=\frac{\exp (-D_{ij}^2)}{\sum _{k \ne l}\exp (-D_{lk}^2)}\qquad \text {where } D_{ij}^2=\left\|  \mathbf{x}^{(i)}-\mathbf{x}^{(j)} \right\| ^2,\, \, i\neq j
$$

where the denominator sums over all distinct pairs of data points. Notice the symmetry $p_{ij}=p_{ji}$; hence we can restrict to indices where $i < j$, and the above definition turns to:

$$
p_{ij}=\frac{\exp (-D_{ij}^2)}{\sum _{k < l}\exp (-D_{lk}^2)}\qquad \text {for } \, i < j.
$$

The set of all $p_{ij}$ defines the pmf of probability distribution **P** on all pairs of points in p-dimensional space. The shape of the Gaussian distribution ensures that pairs that are close together are given much more weight than pairs that are far apart.

**Example 0:**

If a data set had only 2 data points $\mathbf{x}^{(1)}, \mathbf{x}^{(2)}$ at distance $d$ in p-dimensional space, then using the definition above:

$$
p_{ij}=\frac{\exp (-D_{ij}^2)}{\sum _{k < l}\exp (-D_{lk}^2)}\qquad \text {where } \, i < j.
$$

we get:

$$
p_{12}=\frac{\exp (-d^2)}{\exp (-d^2)}=1
$$

regardless of the distance $d$, since there is only 1 pair of distinct data points.

- 3. **In Low Dimensions**: Do the same as above in q-dimensional target space. That is, define for each point $\mathbf{y}^{(i)}$ the probability of $\mathbf{y}^{(j)}$ being its neighbor to be:

$$
q_{ij}=\frac{\exp (-\left\|  \mathbf{y}^{(i)} - \mathbf{y}^{(j)} \right\| ^2)}{\sum _{k < l}\exp (-\left\|  \mathbf{y}^{(k)} - \mathbf{y}^{(l)} \right\| ^2)} \qquad (i\neq j).
$$

The set of all $q_{ij}$ define the pmf of a probability distribution **Q** on all pairs of points in the q-dimensional target space.

- 4. **Minimization**: Find points $\{ \mathbf{y}^{(i)}\}$ in the q-dimensional target space that minimizes the Kullback-Leibler (KL) divergence between the probability distributions **P** and **Q**:

$$
\text {KL}(\mathbf{P}||\mathbf{Q}) = \sum _{i \ne j} p_{ij}\log \frac{p_{ij}}{q_{ij}}
$$

where $p_{ij}$ and $q_{ij}$ give the pmfs of **P** and **Q** respectively. In practice, this minimization is implemented using gradient descent methods.

**Remark**: For simplicity, in the definitions of the distributions **P** and **Q**, we have used Gaussian distributions with the same variance at each data point, and simplified the definition of $p_{ij}$ and $q_{ij}$. Practical algorithms are more sophisticated: they use Gaussian distributions with different variances $\sigma ^2_{i}$ at different data points, and then symmetrize between points $i$ and $j$ to get the pmf $p_{ij}$. (See for example [Wikipedia page on T-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)). The different variances $\sigma ^2_{i}$ are chosen to match with user-defined hyperparameters such as $perplexity$.

<center><img src="Images\EX2029.png" alt="Alt Text" width="650" height="500"></center>
<center><img src="Images\EX2030.png" alt="Alt Text" width="650" height="350"></center>
<center><img src="Images\EX2031.png" alt="Alt Text" width="650" height="350"></center>
<center><img src="Images\EX2032.png" alt="Alt Text" width="650" height="400"></center>
<center><img src="Images\EX2033.png" alt="Alt Text" width="650" height="500"></center>
<center><img src="Images\EX2034.png" alt="Alt Text" width="650" height="300"></center>
<center><img src="Images\EX2035.png" alt="Alt Text" width="650" height="500"></center>
<center><img src="Images\EX2036.png" alt="Alt Text" width="650" height="550"></center>
<center><img src="Images\EX2037.png" alt="Alt Text" width="650" height="550"></center>

One widely-used variation of SNE is the **t-distributed stochastic neighbor embedding (t-SNE)**, which uses the t-distribution instead of the Gaussian distribution to define the probability distribution of neigbors in the low-dimensional target space. That is:

$$
q_{ij}=\frac{1/\left(1+\lVert \mathbf{y}^{(i)} - \mathbf{y}^{(j)} \rVert ^2\right)}{\sum _{k < l}1/\left(1+\lVert \mathbf{y}^{(k)} - \mathbf{y}^{(l)} \rVert ^2\right)}.
$$

(This is again a simplified definition)

The heavy tail of the t-distribution reduces the phenomenon of “data points crowding in the middle."

<center><img src="Images\EX2038.png" alt="Alt Text" width="650" height="500"></center>
<center><img src="Images\EX2039.png" alt="Alt Text" width="650" height="550"></center>
<center><img src="Images\EX2040.png" alt="Alt Text" width="650" height="550"></center>

## 4. Classification

Given data $X^{(1)}, X^{(2)},\ldots , X^{(n)}$ that take value in $\mathbb {R}^ p$. A classification problem is a problem in which we look for a function:

$$
f: \mathbb {R}^ p\to \mathcal{C} \qquad \text {where } \mathcal{C}=\{ C_1,\ldots C_ k\}
$$
 
that maps the sample space of the data into a set of class labels $\mathcal{C}=\{ c_1,\ldots c_ m\}$. This function is called a classifer.

The goal of a classification problem is to predict the class that an unseen data point belongs to. In this lecture, we will discuss classification methods in a supervised learning setting, where data with true class labels are available.

**Methods of Classification**

We will begin with methods with the most model assumptions and proceed to those with least assumptions:

- Classification using Bayes Rule
- Quadratic discriminant analysis (QDA) and Linear discriminant analysis (LDA)
- Logistic regression
- Support vector machine (SVM).

Neural networks are also commonly used for classification. They are complex models defined by many parameters. Counter to intuition in classical statistics where overfitting generally occurs as models become too complex, neural networks generalize very well to unseen data, exactly due to the super over-parameterization of the model. We will not discuss neural networks.