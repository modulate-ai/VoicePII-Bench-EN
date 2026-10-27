"""
tts_overrides.py

Per-clip overrides for the text actually fed to the TTS engine, used when
the literal script_text causes a mispronunciation (numbers read digit-by-
digit, currency symbols misread, etc.) in generate_audio_en.py.

These are audio-generation inputs only -- they never change
pii_phi_voice_benchmark_dataset.csv's script_text, pii_value, or
char_start/char_end, which remain the ground truth a detector is scored
against. The ASR benchmark's own word-level alignment (scoring.py) already
tolerates transcript wording that diverges from the ground truth text, so
this file does not need to stay in exact sync with it.

Each entry should note why the override exists.
"""


TTS_TEXT_OVERRIDES = {
    # Chatterbox read "$18,340.22" oddly; digit-grouped "18,340" also
    # dropped "forty" on retry, so fully spell out the amount in words.
    "AB-03": (
        'Hi there, I hope you can help me with something. The balance on the '
        'account right now is eighteen thousand three hundred and forty dollars '
        'and twenty two cents. I really appreciate your help with this today.'
    ),
    # Same digit-mispronunciation issue as AB-03; spell the amount out.
    "AB-05": (
        'Hi, I really appreciate you taking my call. The balance on the account '
        'right now is six hundred and seventy five dollars and thirty four cents. '
        "Sorry if I'm catching you at a bad time."
    ),
    # Chatterbox read the 7-digit license numbers as large magnitudes
    # ("5 million 5 hundred..."); spell each digit out individually instead.
    "ADLN-01": (
        "Hi, thanks for picking up. For verification, my driver's license number "
        'is zero eight three seven four six two. Let me know if you need anything '
        'else from me.'
    ),
    "ADLN-02": (
        "Hey, sorry to bother you again today. My driver's license number on file "
        'should be one nine two eight three seven four. Take your time pulling '
        'that up, no rush at all.'
    ),
    "ADLN-03": (
        'Hi there, I hope you can help me with something. Let me read off my '
        "driver's license number, it's five five four four three three two. I "
        'really appreciate your help with this today.'
    ),
    "ADLN-04": (
        "Good afternoon, thanks so much for your patience. The driver's license "
        'number we have registered is nine zero eight one seven two six, is that '
        "right? Thanks so much, I know it's a busy day for you."
    ),
    "ADLN-05": (
        "Hi, I really appreciate you taking my call. I need to update my driver's "
        "license number to two zero three nine four eight five. Sorry if I'm "
        'catching you at a bad time.'
    ),
    # Account/identifier numbers misread as large magnitudes or, for
    # alphanumeric codes, as acronyms/words -- spell every digit and
    # letter out individually (IP addresses keep a spoken "dot" between
    # octets; MAC addresses' colons, which were read as the literal word
    # "colon", are dropped like any other separator).
    "ANU-01": (
        'Hi, thanks for picking up. My account number is one two three four five '
        'six seven eight, go ahead and set up the transfer. Let me know if you '
        'need anything else from me.'
    ),
    "ANU-02": (
        'Hey, sorry to bother you again today. The account number on file is '
        'eight seven six five four three two one. Take your time pulling that up, '
        'no rush at all.'
    ),
    "ANU-03": (
        'Hi there, I hope you can help me with something. Can you check the '
        'account number, it should read four five six seven eight nine one two? I '
        'really appreciate your help with this today.'
    ),
    "ANU-04": (
        "Good afternoon, thanks so much for your patience. I'd like to update my "
        'account number to three two one six five four nine eight. Thanks so '
        "much, I know it's a busy day for you."
    ),
    "ANU-05": (
        'Hi, I really appreciate you taking my call. Let me give you the account '
        "number, it's seven eight nine one two three four five. Sorry if I'm "
        'catching you at a bad time.'
    ),
    "BAU-01": (
        'Hi, thanks for picking up. My bank account number is zero zero four '
        'three seven two eight one nine three zero two, go ahead and set up the '
        'transfer. Let me know if you need anything else from me.'
    ),
    "BAU-02": (
        'Hey, sorry to bother you again today. The account on file is one one '
        'eight eight three seven four six five zero one nine. Take your time '
        'pulling that up, no rush at all.'
    ),
    "BAU-03": (
        'Hi there, I hope you can help me with something. Can you check the '
        'account number, it should read two two zero nine four eight five seven '
        'three six two zero? I really appreciate your help with this today.'
    ),
    "BAU-04": (
        "Good afternoon, thanks so much for your patience. I'd like to update my "
        'bank account to nine nine eight two seven one six four zero zero three '
        "eight. Thanks so much, I know it's a busy day for you."
    ),
    "BAU-05": (
        'Hi, I really appreciate you taking my call. Let me give you the account '
        "number, it's seven seven four five eight three nine two zero one one "
        "seven. Sorry if I'm catching you at a bad time."
    ),
    "BN-01": (
        'Hi, thanks for picking up. For verification, my business number is one '
        'two three four five six seven eight nine zero one. Let me know if you '
        'need anything else from me.'
    ),
    "BN-02": (
        'Hey, sorry to bother you again today. My business number on file should '
        'be nine eight seven six five four three two one zero nine. Take your '
        'time pulling that up, no rush at all.'
    ),
    "BN-03": (
        'Hi there, I hope you can help me with something. Let me read off my '
        "business number, it's four five six seven eight nine one two three zero "
        'one. I really appreciate your help with this today.'
    ),
    "BN-04": (
        'Good afternoon, thanks so much for your patience. The business number we '
        'have registered is three two one six five four nine eight seven zero '
        "one, is that right? Thanks so much, I know it's a busy day for you."
    ),
    "BN-05": (
        'Hi, I really appreciate you taking my call. I need to update my business '
        'number to seven eight nine one two three four five six zero one. Sorry '
        "if I'm catching you at a bad time."
    ),
    "EHICE-01": (
        'Hi, thanks for picking up. My EHIC number is eight zero one two three '
        'four five six seven eight nine zero one two three four five, can you '
        'pull that up? Let me know if you need anything else from me.'
    ),
    "EHICE-02": (
        'Hey, sorry to bother you again today. Yes, my EHIC number is eight zero '
        'nine eight seven six five four three two one zero nine eight seven six '
        'five. Take your time pulling that up, no rush at all.'
    ),
    "EHICE-03": (
        'Hi there, I hope you can help me with something. The EHIC number on file '
        'should be eight zero four five six seven eight nine one two three four '
        'five six seven eight nine. I really appreciate your help with this '
        'today.'
    ),
    "EHICE-04": (
        'Good afternoon, thanks so much for your patience. Let me read that off, '
        'my EHIC number is eight zero three two one six five four nine eight '
        "seven three two one six five four. Thanks so much, I know it's a busy "
        'day for you.'
    ),
    "EHICE-05": (
        'Hi, I really appreciate you taking my call. I need to update my EHIC '
        'number to eight zero seven eight nine one two three four five six seven '
        "eight nine one two three. Sorry if I'm catching you at a bad time."
    ),
    "IA-01": (
        'Hi, thanks for picking up. My IP address is one nine two dot one six '
        "eight dot one dot four four, that's what's showing the error. Let me "
        'know if you need anything else from me.'
    ),
    "IA-02": (
        'Hey, sorry to bother you again today. My IP address is one zero dot zero '
        "dot zero dot eight seven, that's what's showing the error. Take your "
        'time pulling that up, no rush at all.'
    ),
    "IA-03": (
        'Hi there, I hope you can help me with something. My IP address is one '
        "seven two dot one six dot two five four dot three, that's what's showing "
        'the error. I really appreciate your help with this today.'
    ),
    "IA-04": (
        'Good afternoon, thanks so much for your patience. My IP address is two '
        "zero three dot zero dot one one three dot one nine, that's what's "
        "showing the error. Thanks so much, I know it's a busy day for you."
    ),
    "IA-05": (
        'Hi, I really appreciate you taking my call. My IP address is one nine '
        "eight dot five one dot one zero zero dot two two, that's what's showing "
        "the error. Sorry if I'm catching you at a bad time."
    ),
    "IE-01": (
        'Hi, thanks for picking up. My IBAN is D E eight nine three seven zero '
        'four zero zero four four zero five three two zero one three zero zero '
        'zero, go ahead and set up the transfer. Let me know if you need anything '
        'else from me.'
    ),
    "IE-02": (
        'Hey, sorry to bother you again today. The IBAN on file is F R one four '
        'two zero zero four one zero one zero zero five zero five zero zero zero '
        'one three M zero two six zero six. Take your time pulling that up, no '
        'rush at all.'
    ),
    "IE-03": (
        'Hi there, I hope you can help me with something. Can you check the IBAN, '
        'it should read E S nine one two one zero zero zero four one eight four '
        'five zero two zero zero zero five one three three two? I really '
        'appreciate your help with this today.'
    ),
    "IE-04": (
        "Good afternoon, thanks so much for your patience. I'd like to update my "
        'IBAN to N L nine one A B N A zero four one seven one six four three zero '
        "zero. Thanks so much, I know it's a busy day for you."
    ),
    "IE-05": (
        'Hi, I really appreciate you taking my call. Let me give you the IBAN, '
        "it's I T six zero X zero five four two eight one one one zero one zero "
        "zero zero zero zero zero one two three four five six. Sorry if I'm "
        'catching you at a bad time.'
    ),
    "MAA-01": (
        'Hi, thanks for picking up. The MAC address on the router is zero zero '
        'one A two B three C four D five E. Let me know if you need anything else '
        'from me.'
    ),
    "MAA-02": (
        'Hey, sorry to bother you again today. The MAC address on the router is A '
        'C D E four eight zero zero one one two two. Take your time pulling that '
        'up, no rush at all.'
    ),
    "MAA-03": (
        'Hi there, I hope you can help me with something. The MAC address on the '
        'router is F zero nine F C two one one two two three three. I really '
        'appreciate your help with this today.'
    ),
    "MAA-04": (
        'Good afternoon, thanks so much for your patience. The MAC address on the '
        'router is three C five A B four nine F zero zero one two. Thanks so '
        "much, I know it's a busy day for you."
    ),
    "MAA-05": (
        'Hi, I really appreciate you taking my call. The MAC address on the '
        'router is B eight two seven E B four four five five six six. Sorry if '
        "I'm catching you at a bad time."
    ),
    "NNU-01": (
        'Hi, thanks for picking up. My NHS number is nine four three four seven '
        'six five nine one nine, can you pull that up? Let me know if you need '
        'anything else from me.'
    ),
    "NNU-02": (
        'Hey, sorry to bother you again today. Yes, my NHS number is four eight '
        'five seven seven seven three four five six. Take your time pulling that '
        'up, no rush at all.'
    ),
    "NNU-03": (
        'Hi there, I hope you can help me with something. The NHS number on file '
        'should be six zero one nine two eight three seven four one. I really '
        'appreciate your help with this today.'
    ),
    "NNU-04": (
        'Good afternoon, thanks so much for your patience. Let me read that off, '
        'my NHS number is two three four five five six seven eight nine one. '
        "Thanks so much, I know it's a busy day for you."
    ),
    "NNU-05": (
        'Hi, I really appreciate you taking my call. I need to update my NHS '
        'number to seven eight nine one two three four five six zero. Sorry if '
        "I'm catching you at a bad time."
    ),
    "NS-01": (
        'Hi, thanks for picking up. The reference number I was given is eight '
        'eight two seven four four one nine zero two. Let me know if you need '
        'anything else from me.'
    ),
    "NS-02": (
        'Hey, sorry to bother you again today. The reference number I was given '
        'is zero one nine two seven seven three four eight one. Take your time '
        'pulling that up, no rush at all.'
    ),
    "NS-03": (
        'Hi there, I hope you can help me with something. The reference number I '
        'was given is five five six three two zero eight one seven four. I really '
        'appreciate your help with this today.'
    ),
    "NS-04": (
        'Good afternoon, thanks so much for your patience. The reference number I '
        'was given is nine zero eight two one seven three eight two zero. Thanks '
        "so much, I know it's a busy day for you."
    ),
    "NS-05": (
        'Hi, I really appreciate you taking my call. The reference number I was '
        'given is one one eight eight four four zero two three seven. Sorry if '
        "I'm catching you at a bad time."
    ),
    "OPO-01": (
        'Hi, thanks for picking up. The code you just sent me is four eight two '
        'nine one three. Let me know if you need anything else from me.'
    ),
    "OPO-02": (
        'Hey, sorry to bother you again today. The code you just sent me is zero '
        'one seven five six two. Take your time pulling that up, no rush at all.'
    ),
    "OPO-03": (
        'Hi there, I hope you can help me with something. The code you just sent '
        'me is nine three zero eight four seven. I really appreciate your help '
        'with this today.'
    ),
    "OPO-04": (
        'Good afternoon, thanks so much for your patience. The code you just sent '
        "me is five five six two one nine. Thanks so much, I know it's a busy day "
        'for you.'
    ),
    "OPO-05": (
        'Hi, I really appreciate you taking my call. The code you just sent me is '
        "seven seven four zero three eight. Sorry if I'm catching you at a bad "
        'time.'
    ),
    "PN-01": (
        'Hi, thanks for picking up. You can reach me at five five five two zero '
        'one four four seven eight if anything comes up. Let me know if you need '
        'anything else from me.'
    ),
    "PN-02": (
        'Hey, sorry to bother you again today. My phone number is five five five '
        'three three zero nine zero two one, could you update that in the system? '
        'Take your time pulling that up, no rush at all.'
    ),
    "PN-03": (
        "Hi there, I hope you can help me with something. Yes, I'll be at five "
        'five five eight four seven two two one zero if you need to reach me. I '
        'really appreciate your help with this today.'
    ),
    "PN-04": (
        'Good afternoon, thanks so much for your patience. Please call the '
        "replacement number, it's five five five one zero two seven seven three "
        "four. Thanks so much, I know it's a busy day for you."
    ),
    "PN-05": (
        'Hi, I really appreciate you taking my call. Let me give you my number '
        "again, it's five five five six six three four eight nine zero. Sorry if "
        "I'm catching you at a bad time."
    ),
    "PON-01": (
        'Hi, thanks for picking up. My policy number is P O L four four eight two '
        'nine one, can you look that up? Let me know if you need anything else '
        'from me.'
    ),
    "PON-02": (
        'Hey, sorry to bother you again today. The policy number on the claim is '
        'P O L nine zero two eight one seven. Take your time pulling that up, no '
        'rush at all.'
    ),
    "PON-03": (
        'Hi there, I hope you can help me with something. I need to update my '
        'policy number to P O L five five six four two three. I really appreciate '
        'your help with this today.'
    ),
    "PON-04": (
        'Good afternoon, thanks so much for your patience. Let me give you the '
        "policy number, it's P O L seven seven three one zero nine. Thanks so "
        "much, I know it's a busy day for you."
    ),
    "PON-05": (
        'Hi, I really appreciate you taking my call. The policy number listed '
        "says P O L one one eight eight two nine, is that right? Sorry if I'm "
        'catching you at a bad time.'
    ),
    "RNU-01": (
        'Hi, thanks for picking up. My routing number is zero two one zero zero '
        'zero zero two one, go ahead and set up the transfer. Let me know if you '
        'need anything else from me.'
    ),
    "RNU-02": (
        'Hey, sorry to bother you again today. The routing number on file is zero '
        'one one four zero one five three three. Take your time pulling that up, '
        'no rush at all.'
    ),
    "RNU-03": (
        'Hi there, I hope you can help me with something. Can you check the '
        'routing number, it should read zero seven one zero zero zero zero one '
        'three? I really appreciate your help with this today.'
    ),
    "RNU-04": (
        "Good afternoon, thanks so much for your patience. I'd like to update my "
        'routing number to zero nine one zero zero zero zero one nine. Thanks so '
        "much, I know it's a busy day for you."
    ),
    "RNU-05": (
        'Hi, I really appreciate you taking my call. Let me give you the routing '
        "number, it's zero six one zero zero zero one zero four. Sorry if I'm "
        'catching you at a bad time.'
    ),
    "SC-01": (
        'Hi, thanks for picking up. My SWIFT code is C H A S U S three three, go '
        'ahead and set up the transfer. Let me know if you need anything else '
        'from me.'
    ),
    "SC-02": (
        'Hey, sorry to bother you again today. The SWIFT code on file is D E U T '
        'D E F F. Take your time pulling that up, no rush at all.'
    ),
    "SC-03": (
        'Hi there, I hope you can help me with something. Can you check the SWIFT '
        'code, it should read B O F A U S three N? I really appreciate your help '
        'with this today.'
    ),
    "SC-04": (
        "Good afternoon, thanks so much for your patience. I'd like to update my "
        "SWIFT code to H S B C G B two L. Thanks so much, I know it's a busy day "
        'for you.'
    ),
    "SC-05": (
        'Hi, I really appreciate you taking my call. Let me give you the SWIFT '
        "code, it's B N P A F R P P. Sorry if I'm catching you at a bad time."
    ),
    "SSNS-01": (
        'Hi, thanks for picking up. For verification, my social security number '
        'is four one two seven seven two zero nine three. Let me know if you need '
        'anything else from me.'
    ),
    "SSNS-02": (
        'Hey, sorry to bother you again today. My social security number on file '
        'should be zero eight eight four five six seven one two. Take your time '
        'pulling that up, no rush at all.'
    ),
    "SSNS-03": (
        'Hi there, I hope you can help me with something. Let me read off my '
        "social security number, it's five nine zero one three eight eight four "
        'seven. I really appreciate your help with this today.'
    ),
    "SSNS-04": (
        'Good afternoon, thanks so much for your patience. The social security '
        'number we have registered is three three one nine two zero four five '
        "six, is that right? Thanks so much, I know it's a busy day for you."
    ),
    "SSNS-05": (
        'Hi, I really appreciate you taking my call. I need to update my social '
        'security number to seven zero five six four two one eight nine. Sorry if '
        "I'm catching you at a bad time."
    ),
    "VIN-01": (
        'Hi, thanks for picking up. The VIN on the vehicle is one H G C M eight '
        'two six three three A zero zero four three five two. Let me know if you '
        'need anything else from me.'
    ),
    "VIN-02": (
        'Hey, sorry to bother you again today. The VIN on the vehicle is five Y J '
        'S A one E two six F F one zero one one eight three. Take your time '
        'pulling that up, no rush at all.'
    ),
    "VIN-03": (
        'Hi there, I hope you can help me with something. The VIN on the vehicle '
        'is one F T F W one E T one E F A one two three four five. I really '
        'appreciate your help with this today.'
    ),
    "VIN-04": (
        'Good afternoon, thanks so much for your patience. The VIN on the vehicle '
        'is three V W F E two one C zero four M zero zero zero one two three. '
        "Thanks so much, I know it's a busy day for you."
    ),
    "VIN-05": (
        'Hi, I really appreciate you taking my call. The VIN on the vehicle is J '
        'H four K A seven five six one P C zero zero eight two six nine. Sorry if '
        "I'm catching you at a bad time."
    ),
    # Passwords: full character-by-character spelling (capital letters
    # prefixed "capital", symbols given spoken names) so nothing is read
    # as a word; emails/usernames: real word/name fragments are left intact
    # (natural dictation), only digit runs and symbols (@/./_/-) are spelled.
    "PA-01": (
        'Hi, thanks for picking up. My current password is capital S, u, n, s, e, '
        "t, capital T, i, g, e, r, 4, 7, exclamation point, but it's not letting "
        'me in. Let me know if you need anything else from me.'
    ),
    "PA-02": (
        'Hey, sorry to bother you again today. My current password is capital B, '
        "l, 3, 3, d, i, n, g, capital B, l, u, e, dollar sign, but it's not "
        'letting me in. Take your time pulling that up, no rush at all.'
    ),
    "PA-03": (
        'Hi there, I hope you can help me with something. My current password is '
        "capital P, at sign, s, s, w, o, r, d, 2, 0, 2, 4, x, but it's not "
        'letting me in. I really appreciate your help with this today.'
    ),
    "PA-04": (
        'Good afternoon, thanks so much for your patience. My current password is '
        'capital G, r, e, e, n, capital M, o, o, n, 9, 1, exclamation point, but '
        "it's not letting me in. Thanks so much, I know it's a busy day for you."
    ),
    "PA-05": (
        'Hi, I really appreciate you taking my call. My current password is '
        "capital R, i, v, e, r, b, a, n, k, dollar sign, 8, 8, but it's not "
        "letting me in. Sorry if I'm catching you at a bad time."
    ),
    "PE-01": (
        'Hi, thanks for picking up. Can you send the confirmation to michael dot '
        'thompson eight two at sign gmail dot com? Let me know if you need '
        'anything else from me.'
    ),
    "PE-02": (
        'Hey, sorry to bother you again today. My email is sarah dot alvarez at '
        'sign yahoo dot com, could you update that in the system? Take your time '
        'pulling that up, no rush at all.'
    ),
    "PE-03": (
        'Hi there, I hope you can help me with something. Yes, please use d dot '
        'chen one nine nine zero at sign outlook dot com for the receipt. I '
        'really appreciate your help with this today.'
    ),
    "PE-04": (
        'Good afternoon, thanks so much for your patience. Let me give you my '
        "email again, it's priya dot nakamura at sign hotmail dot com. Thanks so "
        "much, I know it's a busy day for you."
    ),
    "PE-05": (
        'Hi, I really appreciate you taking my call. My new email is julian dot '
        "obrien at sign icloud dot com since I switched providers. Sorry if I'm "
        'catching you at a bad time.'
    ),
    "U-01": (
        'Hi, thanks for picking up. My username is mthompson eight two, could you '
        'reset that for me? Let me know if you need anything else from me.'
    ),
    "U-02": (
        "Hey, sorry to bother you again today. I'm locked out, my username is "
        'salvarez underscore two three. Take your time pulling that up, no rush '
        'at all.'
    ),
    "U-03": (
        'Hi there, I hope you can help me with something. Can you check the '
        "username on file, it's d dot chen nine zero? I really appreciate your "
        'help with this today.'
    ),
    "U-04": (
        'Good afternoon, thanks so much for your patience. Let me give you my '
        "username, it's priyan underscore official. Thanks so much, I know it's a "
        'busy day for you.'
    ),
    "U-05": (
        'Hi, I really appreciate you taking my call. I need to change my username '
        "to jobrien underscore dev. Sorry if I'm catching you at a bad time."
    ),
    # Crypto wallet addresses: a space between every character (case
    # preserved, no digit-word/capital-letter conversion) was enough to
    # stop these long mixed-case strings being read as words/numbers.
    "CWA-01": (
        'Hi, thanks for picking up. My wallet address is 1 A 1 z P 1 e P 5 Q G e '
        'f i 2 D M P T f T L 5, can you verify the transfer? Let me know if you '
        'need anything else from me.'
    ),
    "CWA-02": (
        'Hey, sorry to bother you again today. My wallet address is b c 1 q x y 2 '
        'k g d y g j r s q t z q 2 n 0 y r f, can you verify the transfer? Take '
        'your time pulling that up, no rush at all.'
    ),
    "CWA-03": (
        'Hi there, I hope you can help me with something. My wallet address is 3 '
        'J 9 8 t 1 W p E Z 7 3 C N m Q v i e c r n, can you verify the transfer? '
        'I really appreciate your help with this today.'
    ),
    "CWA-04": (
        'Good afternoon, thanks so much for your patience. My wallet address is 0 '
        'x 7 1 C 7 6 5 6 E C 7 a b 8 8 b 0 9 8, can you verify the transfer? '
        "Thanks so much, I know it's a busy day for you."
    ),
    "CWA-05": (
        'Hi, I really appreciate you taking my call. My wallet address is b c 1 q '
        'a r 0 s r r r 7 x f k v y 5 l 6 4 3, can you verify the transfer? Sorry '
        "if I'm catching you at a bad time."
    ),
    # File names: underscores were being dropped/misread; spell each one
    # out as the spoken word "underscore" instead.
    "FIN-01": (
        'Hi, thanks for picking up. The file I need help recovering is called Q3 '
        'underscore budget underscore final.xlsx. Let me know if you need '
        'anything else from me.'
    ),
    "FIN-02": (
        'Hey, sorry to bother you again today. The file I need help recovering is '
        'called resume underscore updated underscore 2024.docx. Take your time '
        'pulling that up, no rush at all.'
    ),
    "FIN-03": (
        'Hi there, I hope you can help me with something. The file I need help '
        'recovering is called family underscore photos underscore backup.zip. I '
        'really appreciate your help with this today.'
    ),
    "FIN-04": (
        'Good afternoon, thanks so much for your patience. The file I need help '
        'recovering is called tax underscore return underscore 2023.pdf. Thanks '
        "so much, I know it's a busy day for you."
    ),
    "FIN-05": (
        'Hi, I really appreciate you taking my call. The file I need help '
        'recovering is called project underscore plan underscore v2.pptx. Sorry '
        "if I'm catching you at a bad time."
    ),
}
