# Määrittelydokumentti
## Vuorovaikutusverkkojen päätteleminen hiukkasten trajektoriodatasta (korrelaatio, ennustettavuus ja 1D-CNN)

---

## 1. Perustiedot
- **Ohjelmointikieli:** Python  
  - Keskeisimmät paketit: PyTorch (neuroverkko ja opetus), NumPy (simulaatio ja datankäsittely)
- **Muut kielet (vertaisarviointia varten):** TypeScript, Scala ja R
- **Opinto-ohjelma:** Taloustieteen kandidaatti (VTK)
- **Dokumentaation kieli:** Suomi
- **Aihe / kategoria:** Koneoppiminen  
  - Sovellus: fysikaalinen hiukkassimulaatio ja siitä verkon (jousikytkentöjen) päätteleminen.

---

## 2. Harjoitustyön ydin (tiivistettynä)
Työn ydin on toteuttaa ja testata **useampi verkon inferenssialgoritmi** järjestelmälle, jossa hiukkasparit (mitkä ovat suljetussa laatikossa) voivat olla kytkettyjä jousilla, mutta mallille annetaan vain hiukkasten havaittu liike (trajektori). Toteutan (1) korrelaatiopohjaisen menetelmän, (2) ennustettavuuspohjaisen menetelmän (Granger-/VAR-tyyppinen testi) ja (3) neuroverkkopohjaisen (1D-CNN) ja vertaan niiden kykyä palauttaa oikea vuorovaikutusverkko.

Keskeinen algoritminen haaste on tehdä verkon päätteleminen **pelkästään havainnoista** ja osoittaa menetelmien toimivuus huolellisella testauksella. Toteutan myös simulaattorin ja datan generoinnin, jotta “todellinen” verkko on tunnettu ja arviointi on mahdollista.

---

## 3. Ratkaistava ongelma
Simuloidussa 2D-hiukkasjärjestelmässä osa hiukkasista on kytketty toisiinsa jousilla (Hooken laki), osa ei. Käytössä on vain hiukkasten tila ajan funktiona. Tehtävänä on päätellä binäärinen vuorovaikutusverkko:

- **Syöte:** hiukkasten trajektorit (asemat ja nopeudet ajan yli)
- **Tuloste:** vierusmatriisi `Â`, jossa `Â[i,j]=1` jos hiukkasten i ja j välillä on jousikytkentä, muuten 0

Pohjimmiltaan kyseessä on reunaluokitteluongelma (edge / no-edge).

---

## 4. Toteutettavat algoritmit ja tietorakenteet

### 4.1 Simulaattori (datan generointi)
Toteutan hiukkassimulaattorin, jossa:
- N (pienehkö aika- ja muistikompleksisuuden pienetämiseksi) hiukkasta 2D-tilassa
- osa pareista kytketty jousilla
- voimat lasketaan Hooken lain mukaisesti (tarvittaessa vaimennus stabiliteetin parantamiseksi)
- liikeyhtälöt integroidaan (Euler tai Verlet)

**Tietorakenteet:**
- Hiukkasen tila: `(x, y, vx, vy)`
- Trajektoritensori: `X ∈ R^{M×T×N×4}`
- Todellinen verkko: `A ∈ {0,1}^{N×N}`, symmetrinen, `A[i,i]=0`

### 4.2 Algoritmi 1: Korrelaatiopohjainen verkon inferenssi
**Idea:** kytkettyjen hiukkasten liike on usein “yhdessä” → nopeuskomponenttien korrelaatio kasvaa.

- Lasketaan jokaiselle parille `(i,j)` korrelaatiomitta nopeuksista (vx, vy) ajan yli.
- Kynnystetään korrelaatio: `Â_corr[i,j] = 1`, jos korrelaatio ≥ τ, muuten 0.
- Valitaan τ validaatiodatalla maksimoiden F1.

### 4.3 Algoritmi 2: Ennustettavuuspohjainen inferenssi (VAR/Granger-tyyppinen)
**Idea:** jos j vaikuttaa i:hin, j:n historian lisääminen parantaa i:n seuraavan askeleen ennustamista.

Valitaan ikkuna `L` (esim. 5–10). Ennustetaan `v_i(t+1)` kahdella lineaarisella mallilla:
- **Malli A (oma historia):** selittäjinä `v_i(t), …, v_i(t−L+1)`
- **Malli B (oma + j:n historia):** lisäksi `v_j(t), …, v_j(t−L+1)`

Lasketaan virheet `MSE_A` ja `MSE_B` ja parannus `Δ(i,j)=MSE_A−MSE_B`.  
Kynnystetään `Δ(i,j)` (kynnys valitaan validaatiolla) → `Â_pred`.

### 4.4 Algoritmi 3: Neuroverkkopohjainen inferenssi (1D-CNN aikasarjoille)
**Idea:** opetetaan malli tunnistamaan vuorovaikutukselle tyypillisiä paikallisia kuvioita aikasarjasta.

- Muodostetaan paridatasetti: jokaisesta trajektorista otetaan ikkuna pituudella `L`.
- Yhden pariesimerkin syöte: kahden hiukkasen ominaisuudet ajan yli, esim.  
  kanavat = 8: `(x_i,y_i,vx_i,vy_i,x_j,y_j,vx_j,vy_j)` → muoto `[channels, L]`
- Malli: 1D-konvoluutiokerrokset + pooling + MLP + sigmoid (binary probability).
- Opetus: BCE loss.
- Kynnystys validaatiolla → `Â_nn`.

**Huomio oman toteutuksen osuudesta:**  
Käytän PyTorchia optimointiin ja automaattiseen derivointiin, mutta toteutan itse:
- simulaattorin,
- datasetin muodostuksen,
- korrelaatio- ja ennustettavuusmenetelmät,
- CNN-arkkitehtuurin ja koulutussilmukan,
- metriikat ja visualisoinnit (minimaalinen).

---

## 5. Syötteet ja niiden käyttö

### 5.1 Ohjelman syötteet
- Parametrit simulaattorille: `N`, `T`, `dt`, jousien tiheys `p`, jousivakiot (tai niiden jakauma), mahdollinen vaimennus.
- Datan jako: train/val/test.

### 5.2 Datan koko (tavoitearvot)
- `N = 5–8`
- `T = 20–50`
- `L = 5–10` (ikkuna inferenssille)
- `M = 5 000 – 10 000` trajektoria (train/val/test esim. 70/15/15)

Data generoidaan satunnaisilla verkoilla (jokaiselle parille reuna todennäköisyydellä `p`) niin, että sekä reunoja että ei-reunoja esiintyy riittävästi.

---

## 6. Aika- ja tilavaativuudet (O-analyysit)

Olkoon:
- `N` hiukkasten määrä
- `T` aika-askeleet per trajektori
- `M` trajektorien määrä
- `L` ikkunapituus

### 6.1 Simulaatio
- **Aikavaativuus:** `O(M · T · N²)` (naivisti kaikkien parien jousivoimat, N pieni)
- **Tilavaativuus:** `O(M · T · N)` trajektorien tallennus (tai generoidaan pienempinä batcheina)

### 6.2 Korrelaatiomenetelmä
- **Aikavaativuus:** `O(M · T · N²)` (kullekin parille aggregointi ajan yli)
- **Tilavaativuus:** `O(N²)` (keskimääräiset korrelaatiot pareille)

### 6.3 Ennustettavuusmenetelmä (lineaarinen)
- Datasetin muodostus parille: `O(M · T · L)`
- Parikohtainen laskenta kaikille pareille: `O(N² · M · T · L)` (L ollessa pieni)
- **Tilavaativuus:** `O(N²)` (Δ-arvot ja päätelty verkko)

### 6.4 1D-CNN
- Per minibatch: `O(B · L · C)` (C = kanava/piilokoko, vakio valitulla mallilla)
- Kokonaisaika riippuu epochien määrästä, tavoite on pitää malli kevyehkönä, jotta ajaminen onnistuu perus läppärillä riittävän hyvin (N ja L pieni).

---

## 7. Arviointi (testaus- ja vertailusuunnitelma)

### 7.1 Metriikat (verkon palauttaminen)
- **F1-score** (päämittari)
- **Precision** ja **Recall**
- **Accuracy** (tukimittari)

Metriikat lasketaan reunoille vertaamalla `Â` vs `A`.

### 7.2 Visualisointi (oikeellisuuden ja toimivuuden varmistus)
- Heatmap: `A` ja `Â_*` rinnakkain (kunkin menetelmän tulokset)
- Korrelaatiomatriisi ja kynnysvalinnan vaikutus
- 1D-CNN:n tuottama todennäköisyysmatriisi

### 7.3 Yleistymistesti (valinnainen)
- OOD-testi: testataan eri jousiparametrialueella kuin koulutus (esim. jousivakio)
- Raportoidaan F1/PR myös OOD-testille

---

## 8. Laajuus ja eteneminen

### Minimitavoite
1. Simulaattori ja datagenerointi + “ground truth” -verkot  
2. Korrelaatiomenetelmä + kynnystys + F1/PR  
3. Ennustettavuusmenetelmä + kynnystys + F1/PR  
4. 1D-CNN-menetelmä + koulutus + F1/PR  
5. Vertailu ja visualisoinnit (raportointi)

### Mahdollinen lisäosa (vain jos jää aikaa)
- 1-askeleen dynamiikan ennustaminen päätellyn verkon avulla (yksinkertainen message passing -kerros) ja vertailu MLP:hen.  

### Vaiheistus (ohjeellinen)
- Vko 1–2: simulaattori + datagenerointi + visualisoinnit
- Vko 3–4: korrelaatio + mittarit
- Vko 5–6: ennustettavuus + mittarit
- Vko 7–8: 1D-CNN + opetus + evaluointi
- Vko 9: vertailu + dokumentointi (+ mahdollinen lisäosa)

---

## 9. Lähteet (alustus, pitää tarkentaa projektin edetessä)
- Hooken laki ja jousivoimat
- Numeerinen integraatio (Euler / Verlet) 
- PyTorch dokumentaatio (verkot ja opetus)
- NumPy dokumentaatio (taulukkolaskenta ja analyysi)
- (Lisäosaa varten) Kipf & Welling (2016): *Semi-Supervised Classification with Graph Convolutional Networks*
