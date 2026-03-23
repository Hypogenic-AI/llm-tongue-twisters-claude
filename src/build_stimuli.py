"""Build stimulus sets for the tongue twister experiments."""

import csv
import gzip
import json
import random
import os

random.seed(42)

BASE = "/workspaces/llm-tongue-twisters-claude"


def load_glitch_tokens():
    """Load confirmed glitch tokens from GlitchProber ground truth."""
    glitch_tokens = set()
    gt_dir = os.path.join(BASE, "datasets/glitch_tokens_ground_truth")
    for fname in os.listdir(gt_dir):
        if fname.endswith(".csv"):
            with open(os.path.join(gt_dir, fname)) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        token = row[1].strip()
                        # Filter to interesting tokens (not just whitespace/special)
                        if len(token) >= 3 and token.isprintable() and not token.startswith("<"):
                            glitch_tokens.add(token)
    return list(glitch_tokens)


def load_magikarp_tokens():
    """Load strong-verified magikarp tokens from verification data."""
    magikarp_tokens = []
    vdir = os.path.join(BASE, "datasets/magikarp_verifications")
    for fname in os.listdir(vdir):
        if fname.endswith(".jsonl.gz"):
            with gzip.open(os.path.join(vdir, fname), "rt") as f:
                for line in f:
                    obj = json.loads(line)
                    if obj.get("magikarp") == "strong_verified":
                        tok = obj.get("raw_vocab", "")
                        if len(tok) >= 3 and tok.isprintable() and not tok.startswith("<"):
                            magikarp_tokens.append(tok)
    return magikarp_tokens


def build_glitch_token_set(n=50):
    """Select N diverse glitch tokens for testing."""
    all_glitch = list(set(load_glitch_tokens() + load_magikarp_tokens()))
    # Add the famous ones explicitly
    famous = [
        "SolidGoldMagikarp",
        " SolidGoldMagikarp",
        "TheNitromeFan",
        " TheNitromeFan",
        "DragonMaworker",
        "cloneembedaliased",
        " petertodd",
        "rawdownloadcloneembedreportprint",
        "reportprint",
        "embedreportprint",
        "StreamerBot",
        "InstoreAndOnline",
        "ActionCodeHandler",
        " guiActiveUn",
        "exaboralivedire",
        "PsychExpandoExceptionObjectSyntax",
    ]
    for f in famous:
        if f not in all_glitch:
            all_glitch.append(f)

    # Deduplicate and sample
    all_glitch = list(set(all_glitch))
    random.shuffle(all_glitch)

    # Ensure famous ones are included
    selected = [t for t in famous if t in all_glitch]
    remaining = [t for t in all_glitch if t not in selected]
    selected.extend(remaining[: max(0, n - len(selected))])
    return selected[:n]


def build_control_tokens(n=50):
    """Build control tokens — common English words of varying length."""
    controls = [
        "hello", "world", "computer", "science", "language", "beautiful",
        "university", "programming", "algorithm", "database", "function",
        "variable", "structure", "operation", "keyboard", "mountain",
        "elephant", "chocolate", "telephone", "adventure", "butterfly",
        "dangerous", "education", "furniture", "generator", "happiness",
        "important", "knowledge", "landscape", "messenger", "newspaper",
        "objective", "paragraph", "questions", "reference", "signature",
        "transform", "underwear", "volunteer", "waterfall", "xylophone",
        "yesterday", "astronaut", "boulevard", "catalogue", "discovery",
        "economics", "fantastic", "geography", "histogram", "imaginary",
        "juxtapose",
    ]
    return controls[:n]


def build_adversarial_strings():
    """Construct adversarial strings that may confuse tokenizers."""
    return [
        # Improbable bigrams / rare combos
        "ᄋᄋᄋᄋᄋ",  # Korean jamo repeated
        "𝕊𝕠𝕝𝕚𝕕𝔾𝕠𝕝𝕕",  # Mathematical double-struck
        "H̷e̷l̷l̷o̷",  # Combining characters
        "café" + "\u0301" * 5,  # Excessive combining accents
        "a]b[c}d{e",  # Bracket soup
        "\\u0048\\u0065\\u006C\\u006C\\u006F",  # Escaped unicode
        "base64:SGVsbG8gV29ybGQ=",  # Encoded string
        "🏴󠁧󠁢󠁳󠁣󠁴󠁿🏴󠁧󠁢󠁷󠁬󠁳󠁿🏴󠁧󠁢󠁥󠁮󠁧󠁿",  # Flag emoji with tag sequences
        "NULL\x00BYTE",  # Null byte (will send as text)
        "¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿",  # Latin-1 supplement block
        "ﷺﷻ﷽",  # Arabic ligatures
        "⠓⠑⠇⠇⠕",  # Braille
        "˙ollǝH",  # Upside down
        "Ṫ̈ḣ̈ë̤ M̈ä̤ẗ̈r̤̈ï̤ẍ̈",  # Heavily diacriticized
        "zero\u200Bwidth\u200Bjoiner",  # Zero-width spaces
        "right\u200Fto\u200Fleft\u200Fmark",  # RTL marks
        "tabs\there\tand\tthere",  # Tab characters
        "new\nlines\nin\ntext",  # Newlines embedded
        "AAAAAAAAAA" * 10,  # Repetitive (100 A's)
        "the the the the the the the",  # Repeated common words
    ]


def build_misspelled_documents():
    """Build clean and misspelled document pairs for reproduction testing."""
    docs = []

    # Short document 1
    clean1 = (
        "The quick brown fox jumps over the lazy dog. This sentence contains "
        "every letter of the English alphabet at least once. It has been used "
        "as a typing exercise for many years and remains popular today."
    )
    misspelled1 = (
        "The quikc brown fox jumsp over the lzay dog. This sentance contians "
        "every lettre of the Enlgish alphabet at leasst once. It has been usde "
        "as a typnig exercise for mnay years and remians populer today."
    )

    # Short document 2
    clean2 = (
        "Machine learning is a subset of artificial intelligence that focuses "
        "on building systems that learn from data. Instead of being explicitly "
        "programmed, these systems improve their performance through experience."
    )
    misspelled2 = (
        "Machien learning is a sbuset of artifical intellgience that focuess "
        "on biulding systems that laern from data. Instaed of being explictly "
        "programed, these systmes improve thier performance throught experiense."
    )

    # Short document 3 - technical content
    clean3 = (
        "The HTTP protocol defines several request methods to indicate the desired "
        "action to be performed on a resource. GET retrieves data, POST submits "
        "data, PUT replaces a resource, and DELETE removes a resource."
    )
    misspelled3 = (
        "The HTPT protocol definse several reqeust methods to indicaet the desried "
        "action to be preformed on a resouce. GET retreives data, POST submtis "
        "data, PUT replcaes a resorce, and DELET removes a resouce."
    )

    # Medium document
    clean4 = (
        "Natural language processing has undergone a remarkable transformation "
        "in recent years, driven primarily by the development of large language "
        "models based on the transformer architecture. These models, trained on "
        "vast amounts of text data, have demonstrated surprising capabilities "
        "in understanding and generating human language. From simple text "
        "classification to complex reasoning tasks, language models continue "
        "to push the boundaries of what machines can achieve with language. "
        "However, significant challenges remain, including issues of bias, "
        "hallucination, and the fundamental question of whether these models "
        "truly understand the text they process or merely produce statistically "
        "likely continuations."
    )
    misspelled4 = (
        "Natrual language procesing has undergoen a remarkabel transformation "
        "in recetn years, dirven primarily by the developement of large languaeg "
        "models basde on the transfomer architecture. These modlse, trained on "
        "vast amoutns of text data, have demonstarted suprising capabilites "
        "in undersatnding and generatign human langauge. From smiple text "
        "classificaiton to complxe reasoning taksks, language modlse continue "
        "to psuh the boundareis of waht machines can acheive with langugae. "
        "Howveer, signficant challenges remian, including issuees of bais, "
        "hallucinatoin, and the fundamnetal question of wheather these modlse "
        "truly undersatnd the text they processe or merelry produce statsitically "
        "likelry continuaitons."
    )

    # Long document
    clean5 = (
        "The history of computing is a fascinating journey that spans several "
        "centuries. The earliest computing devices were mechanical calculators, "
        "such as the abacus, which has been used for thousands of years across "
        "many cultures. In the seventeenth century, Blaise Pascal invented the "
        "Pascaline, a mechanical calculator that could perform addition and "
        "subtraction. Later, Gottfried Wilhelm Leibniz created a machine that "
        "could also multiply and divide. These early machines laid the groundwork "
        "for the development of more sophisticated computing devices. "
        "In the nineteenth century, Charles Babbage designed the Analytical Engine, "
        "which is often considered the first general-purpose computer. Although "
        "it was never completed during his lifetime, the design included many "
        "features found in modern computers, such as an arithmetic logic unit, "
        "control flow through conditional branching and loops, and integrated "
        "memory. Ada Lovelace, who worked with Babbage, is often credited as "
        "the first computer programmer for her work on the Analytical Engine. "
        "The twentieth century saw rapid advances in computing technology. "
        "The development of vacuum tubes led to the creation of electronic "
        "computers such as ENIAC, which could perform calculations thousands "
        "of times faster than mechanical devices. The invention of the transistor "
        "in nineteen forty-seven revolutionized electronics and made computers "
        "smaller, faster, and more reliable. The integrated circuit, developed "
        "in the late nineteen fifties, further miniaturized electronic components "
        "and paved the way for the microprocessor revolution of the nineteen seventies."
    )
    misspelled5 = (
        "The hisotry of computing is a fascianting journey that sapns several "
        "centuires. The earliest computign devices were mechancial calculators, "
        "such as the abacsu, which has been usde for thosands of years accross "
        "many cultrues. In the seventeenth centruey, Blaise Pasacl invented the "
        "Pascalnie, a mechancial calculator that could preform additon and "
        "subtarction. Later, Gottfreid Wilhelm Leibnzi created a machine that "
        "could aslo multiply and divdie. These earyl machines laid the grondwork "
        "for the developement of more sophistacated computing devicse. "
        "In the ninteenth century, Chrales Babbage desigend the Analytical Enigne, "
        "which is often considred the first general-purpose compueter. Although "
        "it was never compleeted during his lifetme, the desgn included many "
        "features found in modrn computers, such as an arithmetic lgoci unit, "
        "contrl flow through condtional branching and lopps, and intergrated "
        "memory. Ada Lovelcae, who worked with Babbage, is often credtied as "
        "the first computer programer for her work on the Analytical Enigne. "
        "The twentienth century saw rapdi advances in computing technolgy. "
        "The developement of vacuume tubes led to the creatoin of electronic "
        "computers such as ENIAC, which could preform calculations thosands "
        "of times fastre than mechancial devices. The invetnion of the transistor "
        "in ninteen forty-seven revolutonized electronics and made computers "
        "smallre, fastre, and more relialbe. The intergrated circuit, develoeped "
        "in the late ninteen fifties, furhter miniaturized electrnoic components "
        "and paved the way for the microproccesor revolutoin of the ninteen seventise."
    )

    docs.append({"name": "short_general", "clean": clean1, "misspelled": misspelled1, "length": "short"})
    docs.append({"name": "short_technical_ml", "clean": clean2, "misspelled": misspelled2, "length": "short"})
    docs.append({"name": "short_technical_http", "clean": clean3, "misspelled": misspelled3, "length": "short"})
    docs.append({"name": "medium_nlp", "clean": clean4, "misspelled": misspelled4, "length": "medium"})
    docs.append({"name": "long_computing_history", "clean": clean5, "misspelled": misspelled5, "length": "long"})

    return docs


def build_all_stimuli():
    """Build and save all stimulus sets."""
    stimuli = {
        "glitch_tokens": build_glitch_token_set(50),
        "control_tokens": build_control_tokens(50),
        "adversarial_strings": build_adversarial_strings(),
        "documents": build_misspelled_documents(),
    }

    out_path = os.path.join(BASE, "results/stimuli.json")
    with open(out_path, "w") as f:
        json.dump(stimuli, f, indent=2, ensure_ascii=False)

    print(f"Saved stimuli to {out_path}")
    print(f"  Glitch tokens: {len(stimuli['glitch_tokens'])}")
    print(f"  Control tokens: {len(stimuli['control_tokens'])}")
    print(f"  Adversarial strings: {len(stimuli['adversarial_strings'])}")
    print(f"  Documents: {len(stimuli['documents'])}")

    # Print some examples
    print("\nSample glitch tokens:", stimuli["glitch_tokens"][:10])
    print("Sample control tokens:", stimuli["control_tokens"][:10])

    return stimuli


if __name__ == "__main__":
    build_all_stimuli()
