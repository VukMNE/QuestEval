context_texts = [
    '... Že naslednji teden naj bi se po desetih letih ponovno začela gradnja najvišje nenaseljene stolpnice na svetu, 597 metrov visokega Goldin Finance 117 v severnem kitajskem pristaniškem mestu Tianjin. ...',
    '... Že naslednji teden naj bi se po desetih letih ponovno začela gradnja najvišje nenaseljene stolpnice na svetu, 597 metrov visokega Goldin Finance 117 v severnem kitajskem pristaniškem mestu Tianjin. ...'
]

text = """Že naslednji teden naj bi se po desetih letih ponovno začela gradnja najvišje nenaseljene stolpnice na svetu, 597 metrov visokega Goldin Finance 117 v severnem kitajskem pristaniškem mestu Tianjin.
Gradnja nebotičnika Goldin Finance 117, znanega tudi kot China 117 Tower, ki je bil s 117 nadstropji zamišljen kot najvišji na Kitajskem, se je začela leta 2008, po prvotnih načrtih pa naj bi bil dokončan leta 2014.
A gradnjo so zaradi velike recesije leta 2010 začasno ustavili, nato pa dela nadaljevali leta 2011, ko so za nov datum odprtja postavili 2018–2019.
Vrh takrat pete najvišje stavbe na svetu so dokončali 8. septembra 2015, a so se dela zaradi sesutja kitajske borze ponovno ustavila, nepremičninsko podjetje Goldin Properties s sedežem v Hongkongu je šlo v stečaj, stolpnica pa je vse od takrat prepuščena zobu časa.
A po poročanju kitajskih državnih medijev, na katere se sklicuje CNN, naj bi se dela zdaj končno nadaljevala, stolpnica pa naj bi bila zgrajena leta 2027.
Na vrhu bazen in razgledna ploščad
Zgradba je bila zgrajena s t. i. "mega stebri", da bi bila tako zaščitena pred močnim vetrom in potresi, na vrhu v obliki diamanta pa naj bi bil bazen in opazovalna ploščad.
V višjih nadstropjih naj bi bili po načrtih arhitekturnega biroja P&T Group pisarniški uradi in petzvezdični hotel.
Kot je razbrati iz novega gradbenega dovoljenja, v katerem vrednost naložbe ocenjujejo na skoraj 569 milijonov juanov (68 milijonov evrov), bodo iz imena stavbe opustili naziv propadlega podjetja, ni pa jasno, ali se bo spremenila tudi namembnost stolpnice.
V zadnjem desetletju so zapuščeni nebotičniki postali simbol nepremičninskih tegob Kitajske. Leta 2020 sta ministrstvo za infrastrukturo ter nacionalna komisija za razvoj objavila navodila, v katerih sta prepovedala nove stolpnice, višje od 500 metrov, s čimer naj bi tudi zajezili špekulativno poslovanje, pogosto povezano s tovrstnimi projekti.
"""

ners = [
    'Goldin Finance 117',
    '597 metrov'
    # 'leta 2008',
    # 'leta 2010',
    # 'leta 2011',
    # '8. septembra 2015',
    # 'leta 2027',
    # '"mega stebri"',
    # 'pisarniški uradi in petzvezdični hotel',
    # '569 milijonov juanov'
]

questions = [
    'Kako se imenuje najvišja nenaseljena stolpnica na svetu? [END]',
    'Kako visok je Goldin Finance 117? [END]'
    # 'Kdaj se je začela gradnja Goldin Finance 117? [END]',
    # 'Katero leto so začasno ustavili gradnjo? [END]',
    # 'Kdaj so obnovili dela na stolpnici? [END]',
    # 'Kdaj so dokončali vrh stolpnice? [END]',
    # 'Kdaj naj bi bila stolpnica dokončana? [END]',
    # 'S čim je bila zgrajena stolpnica za zaščito pred vetrom? [END]',
    # 'Kaj naj bi bila v višjih nadstropjih stolpnice? [END]',
    # 'Kolikšno vrednost naložbe ocenjujejo v gradbenem dovoljenju? [END]',
]

few_shot_prompts = []


def build_few_shot_prompt_examples():
    full_prompt = ''
    for q, a, c in zip(questions, ners, context_texts):
        cont = c.replace(a, '<ans>' + a + '</ans>')
        prompt =  (
            f"Besedilo: {cont}\n"
            f"Odgovor: {a}\n"
            f"Vprašanje: {q}\n\n\n"
        )

        full_prompt += prompt

    return full_prompt

print(build_few_shot_prompt_examples())
