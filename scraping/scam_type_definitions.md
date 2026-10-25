# Scam type definitions

How the `scam_type` column in `data/antifakenews_scam.csv` was labelled. The five types are the
Callas scam types, with wider definitions so that every scam on antifakenewscenter.com fits one.

`llm_labeler.py` sends this whole file to Qwen as its instructions, so a change here changes the labels.

## Types

| Type | Definition |
|---|---|
| `fake_authority` | Pretends to be a trusted organization or person in an official role (government agency, police, bank, utility, post or courier, company or shop), often through a fake page or account in its name. It threatens the victim, or offers a service such as a loan, a driving licence, money recovery or a job, then demands a fee, fine or transfer. |
| `romance` | Builds trust as a partner, friend, relative or acquaintance, or takes over or copies the account of someone the victim knows or a celebrity, then asks to borrow money or for a transfer. |
| `investment` | Promises high or guaranteed returns from stocks, crypto, trading, funds or paid "task" jobs. Shows fake profits, then blocks withdrawals until more "fees" are paid. |
| `phishing_otp` | Steals OTPs, passwords, personal or account data, or control of the device, through a link, SMS, QR code, fake website, or an app the victim is told to install, including remote-control apps. |
| `lottery` | Claims the victim won or is entitled to a prize, gift, cash handout, welfare payment or lottery tip, then asks for a fee or bank details before they can receive it. |

## Tie-break

When a post fits several types, the first match in this order wins:

1. `romance`: the scam relies on a personal relationship or trust in someone the victim knows.
2. `investment`: the lure is profit or returns.
3. `lottery`: the lure is a prize, something free, or an entitlement.
4. `phishing_otp`: the main harm is stolen credentials, data or device control.
5. `fake_authority`: everything else that uses impersonation to get paid.

## Scam method fields

Three columns of the dataset describe how each scam works, in short Thai phrases.

| Column | What it holds | Example (post 78875) |
|---|---|---|
| `method_channel` | Where it happens, from a fixed list, comma separated | เพจเฟซบุ๊ก |
| `method_scammer_action` | The story or offer used to hook the victim | เปิดสอนเทรดหุ้นและหลักสูตรพื้นฐานตลาดการเงิน |
| `method_victim_action` | What the victim is pushed to do | ถูกชักชวนให้ลงทุน |

Channel values: เพจเฟซบุ๊ก, กลุ่มเฟซบุ๊ก, Messenger, TikTok, LINE, SMS, โทรศัพท์, เว็บไซต์, อีเมล, เอกสาร,
แอปพลิเคชัน, โซเชียลมีเดีย (social media, platform not named), พอดแคสต์, QR Code, Google Form.