# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from langchain_core.prompts import PromptTemplate


# PROMPT = PromptTemplate.from_template(
# '''<task>
# You are an expert call center assistant. You will be given a set of utterances in <utterances> </utterances> tags, each one on a new line.
# The utterances are part of callcenter conversations between the customer and the support agent and are under the same topic.
# Your task is to generate a short theme label describing the topic. 
# The theme label should be under 5 words and describe the desired customer's action in the call.
# The theme label is a verb phrase that begins with a verb which is in its original form (not in "-ing", "-ed" form).


# Examples:
# <theme_label>track order</theme_label>
# <theme_label_explanation>The customer is inquiring about the details of the order.</theme_label_explanation>

# <theme_label>find nearest branch</theme_label>
# <theme_label_explanation>The customer is asking to find nearest branch and its openning hour.</theme_label_explanation>

# <guidance>
# Output your response in the following way.
# <theme_label_explanation>Your short step-by-step explanation behind the theme</theme_label_explanation>
# <theme_label>your theme label</theme_label>
# </guidance>
# </task>

# H:
# <utterances>
# {utterances}
# </utterances>
# '''
# )



PROMPT = PromptTemplate.from_template(
'''<task>
You are an expert call center assistant. You will be given a set of utterances in <utterances> </utterances> tags, each one on a new line.
The utterances are part of callcenter conversations between the customer and the support agent.
Your task is to generate a short label describing the theme of all the given utterances. The theme label should be under 5 words and describe the desired customer's action in the call.

Example:
<theme_label>Technology</theme_label>
<theme_label_explanation>This cluster is related to advancements in technology.</theme_label_explanation>

<guidance>
Output your response in the following way.
<theme_label_explanation>Your short step-by-step explanation behind the theme</theme_label_explanation>
<theme_label>your theme label</theme_label>
</guidance>
</task>

H:
<utterances>
{utterances}
</utterances>
'''
)

# SECTION_1_RULES = \
# '''Theme labels exclude unneeded and undesirable words.

# 1. Theme labels should be concise (2--5 words long). They should only include essential words (see Word types and Examples below).
# 2. Essential words will primarily include content (open-class) words. Function (closed-class) words should be excluded.
# 3. Prepositions may be included as needed but should be avoided when there is a synonymous alternative label without a preposition.
# 4. Theme labels should also exclude context-sensitive words like pronouns (him, her, them, it, us, etc.) and demonstratives (this, that, those, etc.).

# Word types

# Content/open-class words:
#     * nouns (items, insurance, information, order, etc.)
#     * main verbs (check, inquire, add, explore, etc.)
#     * adjectives (new patient, missing item, etc.)
#     * other modifying words (shipping information, product options, etc.)

# Function/closed-class words:
#     * articles/determiners (the, a, etc.)
#     * auxiliary verbs (have or be, as in I have eaten or I am eating)
#     * copulas 
#     * negation (not or -n't, as in not on time or didn't arrive)
#     * conjunctions (and, or, but, etc.)
#     * complementizers (clause-embedding uses of that, for, if, whether, because, etc.)
#     * modals (can, could, will, would, may, might, must, shall)
#     * question words (who, what, where, when, how, why) 

# Context-sensitive words:
#     * pronouns (she, he, they, it, her, his, etc.)
#     * demonstratives (this, these, that, those, etc.)
#     * temporal adverbs (yesterday, tomorrow, next week, etc.)
#     * other context-sensitive language
#     * one, as in I'm looking for a nearby branch. Can you find one?
#     * deleted nouns (noun ellipsis), as in I found his order, but not yours __.

# Examples

# For a theme covering order tracking:
#     * Good: track order
#     * Good: track shipment
#     * Bad: track an order (includes an article)
#     * Bad: track their order (includes a pronoun)

# For a theme covering finding the nearest branch of a chain:
#     * Good: find nearest branch
#     * Good: find closest branch
#     * Bad: find nearest one (includes context-sensitive one)
#     * Bad: check if there's a nearby branch (includes a complementizer if; includes a form of be)'''

# SECTION_2_RULES = \
# '''Theme labels are verb phrases that classify events.
# 1. A verb phrase begins with a verb and may include arguments or modifiers of the verb (such as a direct object).
# 2. The verb should be in its citation form, lacking any complex morphology such as tense or agreement suffixes.
# 3. The citation form of a verb is what would normally follow the infinitive to, such as sign up in I'd like to sign up.
# 4. Theme labels should not be other phrase types, such as noun phrases.
# 5. The verb phrase should describe a class of events.
# Events are things that can be said to happen, unlike states (e.g. learn [event] vs. know [state]), entities (e.g. redeem [event] vs. redemption
# [entity]), properties (e.g. complain [event] vs. angry [property]), and claims (report defect [event] vs. product is defective [claim]).

# Examples

# For a theme covering membership sign-ups:
#     * Good: sign up for membership (verb phrase; describes a kind of signing up event)
#     * Bad: signing up for membership (verb phrase, but verb is not in citation form)
#     * Bad: membership sign-up (noun phrase; describes a kind of entity)
#     * Bad: memberships (noun phrase; describes a kind of entity)

# For a theme covering requests to check in early at a hotel:
#     * Good: request early check-in (verb phrase; describes a kind of requesting event)
#     * Bad: requested early check-in (verb phrase, but verb is not in citation form)
#     * Bad: request for early check-in (noun phrase; describes a kind of entity)
#     * Bad: customer wants early check-in (this is a claim)

# For a theme covering reporting a defective product:
#     * Good: report defective product (verb phrase; describes events)
#     * Bad: reporting defective product (verb phrase, but verb is not in citation form)
#     * Bad: believe product is defective (verb phrase, but describes a state rather than an event)
#     * Bad: defective product (noun phrase; describes a kind of entity)'''

# SECTION_3_RULES = \
# '''Theme labels are informative and actionable yet sufficiently general.
# 1. Theme labels should be informative enough to substantially narrow down the set of possible customer issue resolution steps (the steps to resolve
# the problem/need that drove the customer to make contact). For example, check balance is probably associated with a standard procedure for checking
# the balance of a range of customer account types, but perform check is so broad that it could be associated with an extremely diverse group of issue
# resolutions. Non-actionable theme labels may be excessively vague or uninformative, and hence not very useful.

# Examples

# For a theme covering appointment-scheduling themes:
#     * Good: schedule appointments
#     * Bad: ask about appointments (probably too general)
#     * Bad: schedule appointment for next week (too specific)
#     * Bad: schedule appointment for elderly parent (too specific)

# For a theme covering adding a recognized user to an existing account or policy:
#     * Good: add user
#     * Bad: add one (too general)
#     * Bad: add oldest child (too specific)

# For a theme covering user password issues:
#     * Good: reset password
#     * Good: troubleshoot password
#     * Bad: secure account (too general)
#     * Bad: reset password again (too specific)

# For a theme covering credit or debit card charge disputes:
#     * Good: dispute charge
#     * Bad: complain about charge (too general)
#     * Bad: file card complaint (too general)
#     * Bad: dispute charge for defective blender (too specific)'''
    
# PROMPT = PROMPT.partial(rules_content_1=SECTION_1_RULES, rules_content_2= SECTION_2_RULES, rules_content_3=SECTION_3_RULES)

