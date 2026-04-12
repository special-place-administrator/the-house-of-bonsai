import * as fs from 'fs';
import * as cheerio from 'cheerio';

const html = fs.readFileSync('yahoo.html', 'utf8');
const $ = cheerio.load(html);

$('.algo').each((_, e) => {
    const rawUrl = $(e).find('a').attr('href') || '';
    let realUrl = rawUrl;
    if (rawUrl.includes('/RU=')) {
        const afterRu = rawUrl.split('/RU=')[1];
        const beforeSlash = afterRu.split('/')[0];
        realUrl = decodeURIComponent(beforeSlash);
    }
    const title = $(e).find('h3').text();
    console.log(realUrl);
});
