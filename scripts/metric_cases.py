# Block E: 30 hand-made cases for eval_metrics.py, written BEFORE the metric code (2026-11-14).
# Each: (id, category, headline, article, expect_fabricated, expect_numbers_only, expect_listicle)
GENERIC = ("The council met on Tuesday to discuss the plan. Residents said the change would help families and children. "
           "A local teacher spoke about how the idea started in her classroom and why it matters to everyone in the town. "
           "Many people at the meeting said they had never seen anything like it before.")
CASES = [
 # --- 9 Title-Case teasers, no real entity: NOT fabricated
 (1, "teaser", "This Mayor Did Something Nobody Expected. You Won't Believe What Happened Next", GENERIC, 0, 0, 0),
 (2, "teaser", "Here's Why Everyone Is Talking About This Simple Idea", GENERIC, 0, 0, 0),
 (3, "teaser", "What This Teacher Said Will Make You Rethink Everything", GENERIC, 0, 0, 0),
 (4, "teaser", "The One Thing Nobody Tells You About Saving Money", GENERIC, 0, 0, 0),
 (5, "teaser", "You Won't Believe How This Small Town Changed Its Future", GENERIC, 0, 0, 0),
 (6, "teaser", "Watch What Happens When A Child Asks A Simple Question", GENERIC, 0, 0, 0),
 (7, "teaser", "This Is The Most Important Story You Will Read Today", GENERIC, 0, 0, 0),
 (8, "teaser", "How One Woman Turned Her Life Around In The Most Amazing Way", GENERIC, 0, 0, 0),
 (9, "teaser", "The First Time She Saw Her Home Again Was Unforgettable", GENERIC, 0, 0, 0),
 # --- 4 listicles: NOT fabricated, listicle flag
 (10, "listicle", "7 Reasons This New Law Matters For Every Family", GENERIC, 0, 0, 1),
 (11, "listicle", "5 Things You Should Know About The Housing Crisis", GENERIC, 0, 0, 1),
 (12, "listicle", "10 Ways Cities Are Fighting Climate Change", GENERIC, 0, 0, 1),
 (13, "listicle", "3 Signs The Economy Is Changing Faster Than Expected", GENERIC, 0, 0, 1),
 # --- 5 alias / demonym / surname: NOT fabricated
 (14, "alias", "U.S. Imposes New Tariffs on Chinese Steel",
  "The United States on Monday announced tariffs on steel imported from China, officials in Washington said.", 0, 0, 0),
 (15, "alias", "UK Prime Minister Faces Pressure Over Brexit Talks",
  "Britain's prime minister faced growing pressure on Thursday as Brexit negotiations stalled in London.", 0, 0, 0),
 (16, "alias", "French Voters Head to the Polls",
  "Voters in France went to the polls on Sunday in a closely watched election.", 0, 0, 0),
 (17, "alias", "Merkel Defends Refugee Policy",
  "German Chancellor Angela Merkel on Wednesday defended her government's refugee policy before parliament.", 0, 0, 0),
 (18, "alias", "UN Warns of Famine in Yemen",
  "The United Nations warned on Friday that millions of people in Yemen face famine.", 0, 0, 0),
 # --- 3 number formats: NOT fabricated
 (19, "numfmt", "Two Killed in Highway Crash Near Denver",
  "Police said 2 people died when a truck crashed on the highway outside Denver on Saturday.", 0, 0, 0),
 (20, "numfmt", "Startup Raises $1.5 Million in New Funding",
  "The startup said it had raised $1.5m from investors to expand its team.", 0, 0, 0),
 (21, "numfmt", "Unemployment Falls to 4 Percent in Ohio",
  "Ohio's unemployment rate fell to 4% last month, the state said.", 0, 0, 0),
 # --- 5 supported claim-bearing headlines: NOT fabricated
 (22, "supported", "Toyota Recalls 50,000 Cars Over Airbag Defect",
  "Toyota is recalling about 50,000 cars because of a defect in the airbags, the company said on Tuesday.", 0, 0, 0),
 (23, "supported", "Black Lives Matter Protest Draws Thousands in Chicago",
  "Thousands joined a Black Lives Matter march in Chicago on Sunday, organisers said.", 0, 0, 0),
 (24, "supported", "Netflix Shares Jump 12% After Strong Quarter",
  "Shares of Netflix rose 12 percent on Wednesday after the company reported strong subscriber growth.", 0, 0, 0),
 (25, "supported", "Hurricane Maria Leaves Puerto Rico Without Power",
  "Hurricane Maria knocked out power across Puerto Rico, the island's governor said.", 0, 0, 0),
 # --- 5 genuine fabrications: fabricated
 (26, "fabricated", "Obama Praises New Climate Deal",
  "World leaders on Saturday agreed a new climate deal after two weeks of talks in Paris.", 1, 0, 0),
 (27, "fabricated", "Fire Destroys 300 Homes in California",
  "A wildfire destroyed 120 homes in northern California, fire officials said on Monday.", 1, 1, 0),
 (28, "fabricated", "Apple Unveils New Phone in Tokyo",
  "Apple unveiled its new phone at an event at its headquarters in Cupertino on Tuesday.", 1, 0, 0),
 (29, "fabricated", "Storm Leaves 2 Million Without Power",
  "The storm left about 200,000 homes without electricity across the region.", 1, 1, 0),
 (30, "fabricated", "NFL Star Arrested After Nightclub Fight",
  "An NBA player was arrested early on Sunday after a fight outside a nightclub, police said.", 1, 0, 0),
 (31, "fabricated", "Senator Warren Slams Tax Bill",
  "Democrats in the Senate criticised the tax bill on Thursday, saying it favoured the rich.", 1, 0, 0),
 # --- added 2026-11-14 after finding zipf('trump')=4.86+ (spec filter would drop real names); expectations independent of code
 (32, "fabricated", "Trump Signs New Order on Trade",
  "The White House said on Friday that new trade rules would take effect next month.", 1, 0, 0),
 (33, "supported", "Trump Signs New Order on Trade",
  "President Donald Trump signed an executive order on trade on Friday, the White House said.", 0, 0, 0),
]
POSTPROCESS_CASES = [
 ('"Mayor Unveils Plan"', "Mayor Unveils Plan"),
 ("Headline: Mayor Unveils Plan  ", "Mayor Unveils Plan"),
 ("**Headline:** “Mayor Unveils Plan”", "Mayor Unveils Plan"),
 ("'Mayor Unveils Plan'\n", "Mayor Unveils Plan"),
]
