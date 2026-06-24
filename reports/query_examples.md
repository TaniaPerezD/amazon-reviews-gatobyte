# GATOBYTE — Ejemplos Documentados de Recuperación Semántica

**Modelo:** `paraphrase-multilingual-MiniLM-L12-v2`  
**Índice:** FAISS `IndexFlatIP`  
**Top-K:** 5  
**Queries en español — corpus en inglés (cross-language)**

---

## Ejemplo 1 — "problemas frecuentes con el producto"

| Métrica | Valor |
|---|---|
| Precision@5 | **0.40** |
| MRR | **0.50** |
| Cosine Sim promedio | 0.6632 |
| Latencia | 24.5 ms |

### Fragmentos recuperados

**[1]** Similitud: `0.7073` (Alta) | Rating: 1.0 ★ | Sentiment: `negative` | Categoría: `All Electronics` | ASIN: `B098T12JT4`

> The product did not work correctly

**[2]** Similitud: `0.6827` (Alta) | Rating: 1.0 ★ | Sentiment: `negative` | Categoría: `All Electronics` | ASIN: `B07GX4CFTC`

> El producto no sirve, defectuso

**[3]** Similitud: `0.6548` (Alta) | Rating: 1.0 ★ | Sentiment: `negative` | Categoría: `Computers` | ASIN: `B07T688Q8X`

> This product does not fit as described.

**[4]** Similitud: `0.6405` (Alta) | Rating: 3.0 ★ | Sentiment: `neutral` | Categoría: `Cell Phones & Accessories` | ASIN: `B07ZZ2Z3G4`

> Fast delivery, but Product returned, was defective

**[5]** Similitud: `0.6306` (Alta) | Rating: 1.0 ★ | Sentiment: `negative` | Categoría: `All Electronics` | ASIN: `B07VFT95QW`

> Waste of time and money. Product does not work.

---

## Ejemplo 2 — "opiniones sobre duración de batería"

| Métrica | Valor |
|---|---|
| Precision@5 | **1.00** |
| MRR | **1.00** |
| Cosine Sim promedio | 0.7943 |
| Latencia | 73.0 ms |

### Fragmentos recuperados

**[1]** Similitud: `0.8021` (Alta) | Rating: 4.0 ★ | Sentiment: `positive` | Categoría: `All Electronics` | ASIN: `B00HIUAQHW`

> Battery is great last a long time 8 hours

**[2]** Similitud: `0.7974` (Alta) | Rating: 3.0 ★ | Sentiment: `neutral` | Categoría: `All Electronics` | ASIN: `B075T6L8HH`

> Overall, these are pretty good. My big complaint is battery life. It gets nowhere near the claimed 8 hours. More like 3 to 4. I also have a set of Phaiser BHS-730 which only claim 5 hour battery life. Those seen to last much closer to advertised.

**[3]** Similitud: `0.7947` (Alta) | Rating: 5.0 ★ | Sentiment: `positive` | Categoría: `Amazon Devices` | ASIN: `B00OQVZDJM`

> Love the battery life and size

**[4]** Similitud: `0.7924` (Alta) | Rating: 5.0 ★ | Sentiment: `positive` | Categoría: `Camera & Photo` | ASIN: `B0BF59V2F7`

> Once you get the hang of it works very well, great battery life (30 days +~) depends how many “hits” you get with alerts on, good field of vision, lobe the ability to talk through it to whoever is at your door

**[5]** Similitud: `0.7850` (Alta) | Rating: 2.0 ★ | Sentiment: `negative` | Categoría: `All Electronics` | ASIN: `B005GNRF9K`

> This battery pack has a very poor life span. I have had it about 7 months and when it was new I made one 30 minute phone call (back in July) however now I am lucky to get 5-10 minutes on it, and the last couple minutes is constant 'low battery' chime. If the phone is off the cradle for more than a couple hours I can't even make a call as it says Low Battery on the display. Previous battery packs have had much longer ...

---

## Ejemplo 3 — "defectos de fabricación y materiales"

| Métrica | Valor |
|---|---|
| Precision@5 | **0.80** |
| MRR | **1.00** |
| Cosine Sim promedio | 0.6534 |
| Latencia | 20.0 ms |

### Fragmentos recuperados

**[1]** Similitud: `0.6986` (Alta) | Rating: 2.0 ★ | Sentiment: `negative` | Categoría: `Computers` | ASIN: `B072PZLZ25`

> Several pieces of the plastic were already starting to crack as we pulled it out of the box. The design and choice of materials were poor from a "durability" perspective

**[2]** Similitud: `0.6557` (Alta) | Rating: 2.0 ★ | Sentiment: `negative` | Categoría: `Computers` | ASIN: `B01M6WDPZC`

> Plastic construction is not very good, mine broke after normal use

**[3]** Similitud: `0.6425` (Alta) | Rating: 2.0 ★ | Sentiment: `negative` | Categoría: `Computers` | ASIN: `B07CH8L9QD`

> Very brittle and low quality material. All four corners of the case have cracked and are coming apart.

**[4]** Similitud: `0.6354` (Alta) | Rating: 2.0 ★ | Sentiment: `negative` | Categoría: `Office Products` | ASIN: `B09LC2QD9L`

> Materials are weak. Cheap quality. Broke right away when being moved. Support broke off.

**[5]** Similitud: `0.6346` (Alta) | Rating: 2.0 ★ | Sentiment: `negative` | Categoría: `Car Electronics` | ASIN: `B00XJC0Y48`

> ty of copper and will crack, over time, when subject to vibration or bending such as found in a automobile.<br /><br />Other cables, wires, and fittings not tested as enough information was collected to judge that product deviates far enough from description to pose a hazard to the unaware end user. There are likely other flaws. The insulation had a strange smell when torch testing and may not be flame retardant and/...

---

## Ejemplo 4 — "facilidad de configuración y uso"

| Métrica | Valor |
|---|---|
| Precision@5 | **0.60** |
| MRR | **1.00** |
| Cosine Sim promedio | 0.7446 |
| Latencia | 20.0 ms |

### Fragmentos recuperados

**[1]** Similitud: `0.7772` (Alta) | Rating: 5.0 ★ | Sentiment: `positive` | Categoría: `Amazon Devices` | ASIN: `B07H65KP63`

> Handy to have and easy to set up and use

**[2]** Similitud: `0.7551` (Alta) | Rating: 4.0 ★ | Sentiment: `positive` | Categoría: `Amazon Devices` | ASIN: `B075X8471B`

> Is very easy to use once it is set up

**[3]** Similitud: `0.7389` (Alta) | Rating: 4.0 ★ | Sentiment: `positive` | Categoría: `All Electronics` | ASIN: `B09XYMSZLL`

> Very convenient and can be used normally

**[4]** Similitud: `0.7303` (Alta) | Rating: 5.0 ★ | Sentiment: `positive` | Categoría: `Camera & Photo` | ASIN: `B088CQWPY1`

> Works fine no problem easy to use and set up

**[5]** Similitud: `0.7214` (Alta) | Rating: 5.0 ★ | Sentiment: `positive` | Categoría: `All Electronics` | ASIN: `B07L3XB2B7`

> muy práctico y de fácil manejo

---

## Ejemplo 5 — "ruido y temperatura del dispositivo"

| Métrica | Valor |
|---|---|
| Precision@5 | **1.00** |
| MRR | **1.00** |
| Cosine Sim promedio | 0.6295 |
| Latencia | 19.4 ms |

### Fragmentos recuperados

**[1]** Similitud: `0.6720` (Alta) | Rating: 3.0 ★ | Sentiment: `neutral` | Categoría: `All Electronics` | ASIN: `B08WZCCBQJ`

> It’s starts making a noticeable noise after you plug in a few devices. Has me worried it might overheat one day.

**[2]** Similitud: `0.6379` (Alta) | Rating: 2.0 ★ | Sentiment: `negative` | Categoría: `All Electronics` | ASIN: `B087D9VRGC`

> Product works but emits quite a bit of noise

**[3]** Similitud: `0.6331` (Alta) | Rating: 2.0 ★ | Sentiment: `negative` | Categoría: `Computers` | ASIN: `B081D5HBBW`

> This computer makes a strange sound, and then overheats. The sound, almost like the ocean, is quite loud. We are returning it.

**[4]** Similitud: `0.6038` (Alta) | Rating: 3.0 ★ | Sentiment: `neutral` | Categoría: `Cell Phones & Accessories` | ASIN: `B074Y11C8B`

> It has noise distortion in the sound and don't turn up loud

**[5]** Similitud: `0.6008` (Alta) | Rating: 3.0 ★ | Sentiment: `neutral` | Categoría: `Computers` | ASIN: `B07CT7WMBC`

> I was looking for a quiet cooling fan for my A/V equipments in an enclosed cabinet and came accross the coolerguys product on Amazon website. I read some good reviews on this thread which helped me pull the trigger. The price is reasonable, fast shipping,easy to assemble and work flawlessly. The temperature sensor is a nice feature-you don't have to worry about turning on/off the fan. The only complain I have is the ...

---

## Ejemplo 6 — "relación calidad precio"

| Métrica | Valor |
|---|---|
| Precision@5 | **0.80** |
| MRR | **1.00** |
| Cosine Sim promedio | 0.7740 |
| Latencia | 19.6 ms |

### Fragmentos recuperados

**[1]** Similitud: `0.7915` (Alta) | Rating: 3.0 ★ | Sentiment: `neutral` | Categoría: `Computers` | ASIN: `B07CPHVJ64`

> Good product in provided price

**[2]** Similitud: `0.7811` (Alta) | Rating: 4.0 ★ | Sentiment: `positive` | Categoría: `Amazon Home` | ASIN: `B0C1SS8SPC`

> Esta bien por el precio pero la calidad no es la mejor

**[3]** Similitud: `0.7670` (Alta) | Rating: 3.0 ★ | Sentiment: `neutral` | Categoría: `Car Electronics` | ASIN: `B09QY1BNWW`

> Good price for what is expected

**[4]** Similitud: `0.7659` (Alta) | Rating: 5.0 ★ | Sentiment: `positive` | Categoría: `Computers` | ASIN: `B01HVHIZFG`

> Great price, decent quality, does what it is supposed to.

**[5]** Similitud: `0.7643` (Alta) | Rating: 5.0 ★ | Sentiment: `positive` | Categoría: `Computers` | ASIN: `B08JD4CGCD`

> The quality is really good for the right price

---

## Ejemplo 7 — "pantalla y calidad de imagen"

| Métrica | Valor |
|---|---|
| Precision@5 | **1.00** |
| MRR | **1.00** |
| Cosine Sim promedio | 0.7253 |
| Latencia | 19.4 ms |

### Fragmentos recuperados

**[1]** Similitud: `0.7552` (Alta) | Rating: 1.0 ★ | Sentiment: `negative` | Categoría: `Computers` | ASIN: `B0C2WWCBSG`

> Picture quality is distinctly average. High contrast items (e.g. black text on white background) has slight vertical ghosting which I've not been able to correct. Whereas it's beautifully lightweight, when you push the buttons on the underside for brightness (etc) the whole monitor wobbles on the stand and shifts on the desktop. It's a good price but this is a flimsy monitor with a very average picture quality.

**[2]** Similitud: `0.7299` (Alta) | Rating: 2.0 ★ | Sentiment: `negative` | Categoría: `Camera & Photo` | ASIN: `B08X71T55L`

> The picture quality can be better.

**[3]** Similitud: `0.7163` (Alta) | Rating: 3.0 ★ | Sentiment: `neutral` | Categoría: `All Electronics` | ASIN: `B0BGXS26QL`

> Good quality camera focus excellent picture quality good

**[4]** Similitud: `0.7138` (Alta) | Rating: 3.0 ★ | Sentiment: `neutral` | Categoría: `Computers` | ASIN: `B01AZC3J3M`

> it works, but the picture quality is not that great

**[5]** Similitud: `0.7112` (Alta) | Rating: 2.0 ★ | Sentiment: `negative` | Categoría: `Cell Phones & Accessories` | ASIN: `B01CQK4BGA`

> The size and angles are great but quality is lacking. Screen flashes white intermittently(very distracting) and very low quality image on screen

---

