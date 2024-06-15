description = "This is Matt Hergott’s submission to the NVIDIA Generative AI Agents Contest.\nThis is a demonstration version with few privacy safeguards, so do not submit any personal information into the recording."

transcription_error_msg = '''
Transcription of the audio resulted in an error or no text.
1. If you've stopped using the program, you can exit.
2. Check whether an OpenAI error message was printed in the terminal; your
    account might have usage limitations for the OpenAI transcription API.
3. Consider switching your microphone. The Python executable might not be able 
    to 'activate' a microphone for recording even if Python can recognize
    the correct system default microphone. This can be true even if other
    applications have no problem accessing your microphone. 
'''

react_template='You are given an input text representing a conversation between two or more people. Predict 50 important words that are likely to be used in this conversation. Give the results as a list of words separated by commas. You have access to the following tools:\n\n{tools}\n\nUse the following format:\n\nConversation: the input conversation for which you must find 50 important words\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: 50 important words\n\nBegin!\n\nConversation: {input}\nThought:{agent_scratchpad}'

words=['one',
'get',
'new',
'see',
'two',
'way',
'may',
'day',
'got',
'say',
'man',
'use',
'old',
'end',
'big',
'put',
'lot',
'let',
'set',
'god',
'top',
'yet',
'bad',
'men',
'far',
'job',
'try',
'yes',
'run',
'law',
'war',
'car',
'per',
'ago',
'guy',
'pay',
'win',
'air',
'bit',
'hit',
'due',
'ask',
'saw',
'low',
'age',
'buy',
'red',
'u.s',
'act',
'fun',
'art',
'non',
'six',
'son',
'cut',
'sex',
'boy',
'hot',
'tax',
'eat',
'hey',
'key',
'mom',
'cup',
'led',
'dog',
'oil',
'add',
'met',
'bed',
'die',
'sea',
'sir',
'lol',
'ten',
'box',
'etc',
'eye',
'via',
'gas',
'ass',
'ice',
'kid',
'san',
'sun',
'gun',
'wow',
'dad',
'fit',
'pre',
'bar',
'pro',
'fan',
'gay',
'sit',
'mrs',
'pop',
'ran',
'app',
'fat',
'like',
'time',
'also',
'good',
'know',
'make',
'back',
'want',
'well',
'said',
'much',
'even',
'need',
'work',
'year',
'made',
'take',
'many',
'life',
'last',
'best',
'love',
'home',
'long',
'look',
'used',
'come',
'part',
'find',
'help',
'high',
'game',
'give',
'next',
'must',
'show',
'feel',
'sure',
'team',
'ever',
'keep',
'free',
'away',
'left',
'city',
'days',
'name',
'play',
'real',
'done',
'care',
'week',
'case',
'full',
'live',
'read',
'told',
'four',
'hard',
'mean',
'tell',
'seen',
'stop',
'call',
'head',
'took',
'came',
'side',
'went',
'less',
'line',
'says',
'open',
'shit',
'area',
'face',
'five',
'kind',
'hope',
'news',
'able',
'book',
'post',
'talk',
'fact',
'guys',
'half',
'hand',
'mind',
'body',
'food',
'true',
'fuck',
'lost',
'room',
'else',
'girl',
'john',
'nice',
'yeah',
'york',
'idea',
'past',
'move',
'wait',
'data',
'late',
'stay',
'deal',
'soon',
'turn',
'form',
'fire',
'easy',
'near',
'plan',
'west',
'kids',
'list',
'meet',
'type',
'baby',
'song',
'word',
'gave',
'gets',
'self',
'cost',
'held',
'main',
'road',
'town',
'fine',
'hear',
'rest',
'term',
'wife',
'date',
'goes',
'land',
'miss',
'shot',
'site',
'eyes',
'june',
'club',
'died',
'film',
'knew',
'lead',
'dead',
'hold',
'star',
'test',
'view',
'hour',
'wish',
'gold',
'gone',
'july',
'king',
'bank',
'east',
'park',
'role',
'sent',
'bill',
'cool',
'rate',
'save',
'blue',
'fall',
'fast',
'felt',
'size',
'step',
'page',
'paid',
'upon',
'hate',
'send',
'vote',
'hell',
'lord',
'born',
'damn',
'kill',
'poor',
'code',
'door',
'hair',
'lose',
'pick',
'race',
'seem',
'sign',
'walk',
'loss',
'ones',
'safe',
'army',
'goal',
'huge',
'okay',
'ways',
'base',
'deep',
'mark',
'pass',
'risk',
'ball',
'card',
'dark',
'mine',
'note',
'wall',
'boys',
'fans',
'pain',
'paul',
'rock',
'cold',
'anti',
'beat',
'text',
'join',
'kept',
'sort',
'drop',
'fair',
'feet',
'link',
'sale',
'tour',
'jobs',
'sell',
'sold',
'wide',
'fear',
'lady',
'plus',
'unit',
'hurt',
'rule',
'none',
'ship',
'band',
'cash',
'lack',
'wear',
'luck',
'rich',
'skin',
'thus',
'fish',
'glad',
'grow',
'trip',
'cars',
'laws',
'male',
'spot',
'holy',
'lots',
'shop',
'sick',
'uses',
'cell',
'drug',
'foot',
'hall',
'mass',
'nine',
'heat',
'fell',
'ride',
'would',
'first',
'think',
'could',
'right',
'years',
'going',
'still',
'never',
'world',
'great',
'every',
'state',
'three',
'since',
'thing',
'house',
'place',
'found',
'might',
'money',
'night',
'group',
'women',
'start',
'times',
'today',
'point',
'music',
'power',
'water',
'based',
'small',
'white',
'later',
'order',
'party',
'thank',
'using',
'black',
'makes',
'whole',
'maybe',
'story',
'games',
'least',
'means',
'early',
'local',
'video',
'young',
'court',
'given',
'level',
'often',
'death',
'hours',
'south',
'known',
'large',
'wrong',
'along',
'needs',
'class',
'close',
'comes',
'looks',
'cause',
'happy',
'human',
'woman',
'leave',
'north',
'watch',
'light',
'short',
'taken',
'third',
'among',
'check',
'heart',
'asked',
'child',
'major',
'media',
'phone',
'gonna',
'quite',
'works',
'final',
'front',
'ready',
'bring',
'heard',
'march',
'study',
'clear',
'month',
'words',
'board',
'field',
'seems',
'wants',
'fight',
'force',
'issue',
'price',
'shows',
'space',
'total',
'share',
'april',
'sense',
'weeks',
'break',
'event',
'sorry',
'takes',
'girls',
'guess',
'learn',
'added',
'alone',
'hands',
'movie',
'press',
'tried',
'worth',
'areas',
'books',
'sound',
'value',
'lives',
'round',
'stand',
'stuff',
'david',
'drive',
'green',
'match',
'model',
'trust',
'range',
'trade',
'chief',
'james',
'lower',
'style',
'blood',
'china',
'stage',
'terms',
'title',
'enjoy',
'cover',
'legal',
'seven',
'staff',
'super',
'union',
'began',
'built',
'crazy',
'daily',
'knows',
'paper',
'parts',
'voice',
'whose',
'earth',
'rules',
'offer',
'sleep',
'table',
'truth',
'build',
'cases',
'india',
'piece',
'visit',
'wanna',
'wrote',
'gives',
'river',
'shall',
'speak',
'write',
'album',
'eight',
'funny',
'peace',
'sales',
'spent',
'store',
'track',
'ahead',
'allow',
'brown',
'moved',
'plans',
'radio',
'cross',
'focus',
'loved',
'miles',
'speed',
'jesus',
'extra',
'quick',
'agree',
'clean',
'photo',
'scene',
'spend',
'teams',
'coach',
'costs',
'heavy',
'train',
'claim',
'goals',
'gotta',
'hotel',
'judge',
'lines',
'named',
'brain',
'floor',
'image',
'meant',
'reach',
'civil',
'dance',
'stock',
'trump',
'worst',
'beach',
'ended',
'older',
'calls',
'color',
'dream',
'grand',
'names',
'sweet',
'touch',
'doubt',
'drink',
'feels',
'shown',
'basic',
'carry',
'crime',
'fully',
'japan',
'plant',
'smith',
'texas',
'worse',
'award',
'block',
'lived',
'peter',
'rates',
'avoid',
'catch',
'coast',
'trial',
'truly',
'obama',
'queen',
'stars',
'broke',
'glass',
'prior',
'royal',
'people',
'really',
'around',
'always',
'better',
'little',
'things',
'school',
'family',
'please',
'second',
'number',
'called',
'public',
'system',
'person',
'change',
'enough',
'making',
'states',
'though',
'season',
'trying',
'united',
'course',
'health',
'within',
'thanks',
'others',
'social',
'single',
'become',
'coming',
'office',
'almost',
'taking',
'anyone',
'matter',
'pretty',
'friend',
'saying',
'wanted',
'months',
'series',
'either',
'future',
'police',
'rather',
'reason',
'report',
'living',
'behind',
'market',
'former',
'street',
'london',
'chance',
'father',
'across',
'action',
'moment',
'mother',
'energy',
'played',
'points',
'summer',
'killed',
'strong',
'period',
'record',
'common',
'likely',
'center',
'county',
'couple',
'happen',
'inside',
'issues',
'online',
'player',
'return',
'rights',
'higher',
'member',
'middle',
'needed',
'result',
'answer',
'design',
'policy',
'church',
'longer',
'worked',
'became',
'giving',
'ground',
'source',
'follow',
'amount',
'league',
'groups',
'review',
'cannot',
'looked',
'august',
'attack',
'entire',
'french',
'turned',
'choice',
'events',
'simple',
'simply',
'career',
'figure',
'modern',
'forget',
'listen',
'access',
'europe',
'george',
'recent',
'seeing',
'growth',
'places',
'charge',
'create',
'effect',
'except',
'moving',
'weight',
'double',
'expect',
'island',
'normal',
'credit',
'female',
'nearly',
'region',
'travel',
'beyond',
'forces',
'minute',
'nature',
'unless',
'canada',
'income',
'levels',
'posted',
'safety',
'sounds',
'asking',
'friday',
'search',
'author',
'centre',
'german',
'global',
'leader',
'letter',
'nobody',
'sister',
'annual',
'battle',
'degree',
'france',
'sports',
'stupid',
'active',
'cancer',
'master',
'russia',
'wonder',
'africa',
'effort',
'impact',
'latest',
'passed',
'secret',
'senior',
'spring',
'sunday',
'anyway',
'bought',
'choose',
'direct',
'easily',
'finish',
'indian',
'caught',
'closed',
'damage',
'doctor',
'notice',
'highly',
'winter',
'advice',
'broken',
'caused',
'helped',
'nation',
'prices',
'theory',
'agency',
'camera',
'status',
'claims',
'coffee',
'flight',
'google',
'murder',
'showed',
'accept',
'actual',
'appear',
'eating',
'losing',
'mobile',
'opened',
'placed',
'robert',
'another',
'without',
'someone',
'company',
'thought',
'however',
'getting',
'looking',
'already',
'nothing',
'support',
'believe',
'service',
'country',
'general',
'working',
'friends',
'control',
'problem',
'history',
'several',
'started',
'playing',
'members',
'special',
'fucking',
'million',
'morning',
'whether',
'minutes',
'players',
'talking',
'college',
'current',
'example',
'program',
'process',
'outside',
'instead',
'results',
'running',
'america',
'project',
'account',
'include',
'parents',
'similar',
'perfect',
'english',
'private',
'british',
'present',
'finally',
'society',
'average',
'brought',
'certain',
'medical',
'exactly',
'meeting',
'provide',
'usually',
'reading',
'federal',
'feeling',
'picture',
'central',
'changes',
'england',
'forward',
'science',
'various',
'brother',
'natural',
'october',
'quality',
'amazing',
'related',
'serious',
'article',
'decided',
'january',
'perhaps',
'release',
'website',
'written',
'council',
'foreign',
'michael',
'changed',
'popular',
'systems',
'version',
'writing',
'success',
'towards',
'waiting',
'created',
'missing',
'schools',
'percent',
'allowed',
'culture',
'married',
'officer',
'respect',
'tonight',
'century',
'limited',
'network',
'student',
'capital',
'chinese',
'russian',
'station',
'western',
'content',
'despite',
'husband',
'leading',
'message',
'quickly',
'section',
'contact',
'welcome',
'earlier',
'leaving',
'numbers',
'studies',
'winning',
'episode',
'justice',
'manager',
'ability',
'calling',
'happens',
'subject',
'product',
'regular',
'stories',
'workers',
'anymore',
'growing',
'opening',
'opinion',
'biggest',
'defense',
'reasons',
'weekend',
'awesome',
'clearly',
'imagine',
'protect',
'telling',
'address',
'details',
'disease',
'driving',
'germany',
'greater',
'largest',
'machine',
'overall',
'records',
'reports',
'captain',
'effects',
'explain',
'holding',
'parties',
'reality',
'comment',
'killing',
'primary',
'purpose',
'showing',
'teacher',
'william',
'economy',
'meaning',
'mission',
'weather',
'complex',
'evening',
'freedom',
'highest',
'library',
'located',
'offered',
'putting',
'seconds',
'sitting',
'walking',
'attempt',
'channel',
'finding',
'learned',
'reached',
'receive',
'business',
'anything',
'national',
'actually',
'american',
'children',
'everyone',
'together',
'research',
'remember',
'probably',
'possible',
'question',
'services',
'although',
'building',
'students',
'thinking',
'position',
'happened',
'military',
'personal',
'security',
'industry',
'problems',
'training',
'interest',
'original',
'received',
'director',
'evidence',
'official',
'whatever',
'football',
'property',
'complete',
'economic',
'involved',
'language',
'november',
'decision',
'continue',
'election',
'european',
'increase',
'daughter',
'december',
'hospital',
'starting',
'internet',
'practice',
'followed',
'released',
'district',
'minister',
'products',
'straight',
'february',
'included',
'response',
'specific',
'standard',
'provided',
'recently',
'required',
'tomorrow',
'watching',
'addition',
'pressure',
'campaign',
'previous',
'reported',
'consider',
'positive',
'computer',
'favorite',
'movement',
'designed',
'expected',
'includes',
'material',
'contract',
'families',
'features',
'finished',
'majority',
'physical',
'approach',
'compared',
'fighting',
'learning',
'multiple',
'pictures',
'analysis',
'marriage',
'patients',
'speaking',
'supposed',
'congress',
'directly',
'facebook',
'planning',
'programs',
'comments',
'officers',
'politics',
'produced',
'saturday',
'activity',
'division',
'location',
'standing',
'distance',
'exchange',
'northern',
'powerful',
'something',
'different',
'including',
'following',
'president',
'important',
'community',
'political',
'according',
'available',
'education',
'sometimes',
'beautiful',
'companies',
'questions',
'september',
'countries',
'attention',
'character',
'situation',
'currently',
'financial',
'difficult',
'published',
'australia',
'committee',
'potential',
'treatment',
'beginning',
'certainly',
'described',
'statement',
'announced',
'continued',
'knowledge',
'developed',
'generally',
'secretary',
'insurance',
'seriously',
'direction',
'operation',
'christmas',
'increased',
'literally',
'necessary',
'resources',
'yesterday',
'professor',
'effective',
'agreement',
'challenge',
'christian',
'equipment',
'otherwise',
'executive',
'therefore',
'condition',
'interview',
'religious',
'government',
'everything',
'university',
'understand',
'experience',
'department',
'especially',
'production',
'management',
'technology',
'considered',
'washington',
'conference',
'difference',
'population',
'california',
'completely',
'individual',
'particular',
'throughout',
'absolutely',
'additional',
'conditions',
'collection',
'definitely',
'interested',
'successful',
'australian',
'commercial',
'activities',
'commission',
'associated',
'characters',
'eventually',
'operations',
'protection',
'information',
'development',
'performance',
'association',
'interesting',
'immediately',
'significant',
'opportunity',
'independent',
'competition',
'established',
'responsible',
'environment',
'application',
'relationship',
'professional',
'construction',
'particularly',
'organization',
'international',
'administration',
'nurse',
'medicine',
'emergency',
'breath',
'cat',
'newspaper']