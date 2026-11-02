// import
import wmill from "windmill-client";
import { task, workflow } from 'windmill-client';
import { createHash } from "node:crypto";
import pg from "pg";
import { z } from "zod";

// キャッシュ検索・ベクトルストア検索用のベクトル生成戻り値の型
const CalculateVectorSchema = z.object({
  embeddings: z.array(z.array(z.number()).length(1024))
});

type CalculateVectorResult = z.infer<typeof CalculateVectorSchema>;

// ワークフロー入力キーワードを単語単位に切り分けるタスク戻り値の型
const NormalizeSplitTokensSchema = z.object({
  keywords: z.array(z.string())
});

type NormalizeSplitTokensResult = z.infer<typeof NormalizeSplitTokensSchema>;

// 検索キーワード正規化タスク戻り値の型
type NormalizeCryptoResult = {
  hashBase: string,
  queryHash: string
};

// 検索結果 ( ストア検索 & Web 検索 両方 ) の型
type SearchResult = {
  source: string,
  hit: boolean,
  rows: unknown[]
};

// Searxng 検索結果の型
const SearxngSchema = z.object({
  results: z.array(z.object({
    title: z.string(),
    url: z.string(),
    content: z.string()
  }))
});

type SearxngResult = z.infer<typeof SearxngSchema>;

// Searxng 検索結果の達成評価タスク戻り値の型
const SearxngCompletionJudgeSchema = z.object({
  judge: z.boolean()
});

type SearxngCompletionJudgeResult = z.infer<typeof SearxngCompletionJudgeSchema>;

// Tavily 検索結果の型
const TavilySchema = z.object({
  results: z.array(z.object({
    title: z.string(),
    url: z.string(),
    content: z.string()
  }))
});

type TavilyResult = z.infer<typeof TavilySchema>;

// ワークフロー
export const main = workflow(async (
  original: string, cacheStoreName: string, vectorStoreName: string, keyVersion: string,
  ollamaBaseUrl: string, searxngBaseUrl: string, tavilyBaseUrl: string,
  ragdbResourceName: string, tavilyResourceName: string) => {
  // 入力文字列 ( original ) を正規化する
  // 以下 4 つの戻り値を取得
  //   1. NFKC 正規化 ( normalizeNfkcResult )
  //   2. キーワード配列 ( normalizeSplitTokensResult )
  //   3. キーワード配列つなぎ直し文字列 ( normalizeCryptoResult.hashBase )
  //   4. 3 のハッシュ ( normalizeCryptoResult.queryHash )
  const normalizeNfkcResult: string = await task(normalizeNfkc)(original);
  const normalizeSplitTokensResult: NormalizeSplitTokensResult = await task(normalizeSplitTokens)(ollamaBaseUrl, normalizeNfkcResult);
  const normalizeCryptoResult: NormalizeCryptoResult = await task(normalizeCrypto)(normalizeSplitTokensResult.keywords);

  let found: SearchResult | null = null;
  let storeVector: number[] | null = null;

  const basicCacheSearchResult: SearchResult = await task(basicCacheSearch)(cacheStoreName, normalizeCryptoResult.queryHash, keyVersion, ragdbResourceName);
  // キャッシュ検索
  if (basicCacheSearchResult.hit) found = { source: basicCacheSearchResult.source, hit: basicCacheSearchResult.hit, rows: basicCacheSearchResult.rows };
  // キャッシュ検索 ( ベクトルを使った近似検索 )
  if (!found) {
    storeVector = await task(calculateVector)(ollamaBaseUrl, normalizeCryptoResult.hashBase);
    const vectorCacheSearchResult: SearchResult = await task(vectorCacheSearch)(cacheStoreName, storeVector, keyVersion, ragdbResourceName);
    if (vectorCacheSearchResult.hit) found = { source: vectorCacheSearchResult.source, hit: vectorCacheSearchResult.hit, rows: vectorCacheSearchResult.rows };
  }
  // ベクトルストア検索
  if (!found) {
    const v = await task(calculateVector)(ollamaBaseUrl, normalizeNfkcResult);
    const vectorStoreSearchResult = await task(vectorStoreSearch)(vectorStoreName, v, ragdbResourceName);
    if (vectorStoreSearchResult.hit) found = { source: vectorStoreSearchResult.source, hit: vectorStoreSearchResult.hit, rows: vectorStoreSearchResult.rows };
  }
  // ここまでで回答が得られた場合はアーリーリターンする
  // Searxng での検索乱発や Tavily の利用回数消費を抑制
  if (found) {
    return found;
  }

  // Searxng 検索
  const searxngResult: SearxngResult = await task(searxngSearch)(searxngBaseUrl, normalizeCryptoResult.hashBase);
  // Searxng 検索が入力への回答に値するか確認
  const searxngCompletionJudgeResult: SearxngCompletionJudgeResult = await task(searxngCompletionJudge)(ollamaBaseUrl, normalizeNfkcResult, searxngResult);
  if (searxngCompletionJudgeResult.judge) {
    found = { source: "searxng", hit: searxngCompletionJudgeResult.judge, rows: searxngResult.results };
  }
  else {
    // Tavily 検索
    const tavilyResult: TavilyResult = await task(tavilySearch)(tavilyBaseUrl, normalizeCryptoResult.hashBase, tavilyResourceName);
    found = { source: "tavily", hit: true, rows: tavilyResult.results };
  }

  // キャッシュへ検索結果を保存
  // Searxng が「ヒットなし」でも Tavily で必ずヒットする前提となっているため「storeVector が null でない」だけが条件となっている
  if (storeVector) {
    await task(cacheRegister)(cacheStoreName, keyVersion, normalizeCryptoResult.hashBase, normalizeCryptoResult.queryHash, storeVector, found.rows, found.source, ragdbResourceName);
  }
  return found;
});

async function calculateVector(ollamaBaseUrl: string, arg: string): Promise<number[]> {
  return new VectorCalculation().run(ollamaBaseUrl, arg);
}

async function normalizeNfkc(original: string): Promise<string> {
  return new Normalize().normalizeNfkc(original);
}

async function normalizeSplitTokens(ollamaBaseUrl: string, nfkc: string): Promise<NormalizeSplitTokensResult> {
  return new Normalize().normalizeSplitTokens(ollamaBaseUrl, nfkc);
}

async function normalizeCrypto(keywords: string[]): Promise<NormalizeCryptoResult> {
  return new Normalize().normalizeCrypto(keywords);
}

async function basicCacheSearch(store: string, cacheKey: string, keyVersion: string, ragdbResourceName: string): Promise<SearchResult> {
  return new SearchCacheAndVector().basicCacheSearch(store, cacheKey, keyVersion, ragdbResourceName);
}

async function vectorCacheSearch(store: string, vector: number[], keyVersion: string, ragdbResourceName: string): Promise<SearchResult> {
  return new SearchCacheAndVector().vectorCacheSearch(store, vector, keyVersion, ragdbResourceName);
}

async function vectorStoreSearch(store: string, vector: number[], ragdbResourceName: string): Promise<SearchResult> {
  return new SearchCacheAndVector().vectorStoreSearch(store, vector, ragdbResourceName);
}

async function searxngSearch(searxngBaseUrl: string, hashBase: string): Promise<SearxngResult> {
  return new SearchWeb().searxngSearch(searxngBaseUrl, hashBase);
}

async function searxngCompletionJudge(ollamaBaseUrl: string, nfkc: string, results: SearxngResult): Promise<SearxngCompletionJudgeResult> {
  return new SearchWeb().searxngCompletionJudge(ollamaBaseUrl, nfkc, results);
}

async function tavilySearch(tavilyBaseUrl: string, hashBase: string, tavilyResourceName: string): Promise<TavilyResult> {
  return new SearchWeb().tavilySearch(tavilyBaseUrl, hashBase, tavilyResourceName);
}

async function cacheRegister(store: string, keyVersion: string, hashBase: string, queryHash: string, vector: number[], results: unknown, source: string, ragdbResourceName: string) {
  return new CacheRegister().run(store, keyVersion, hashBase, queryHash, vector, results, source, ragdbResourceName);
}

// ワークフロー全体で使用する関数
function zodParseChat<S extends z.ZodType>(body: unknown, schema: S): z.output<S> {
  const envelope = z.object({
    message: z.object({
      content: z.string()
    })
  }).parse(body);
  const output: unknown = JSON.parse(envelope.message.content);
  return schema.parse(output);
}

function zodParseEmbed<S extends z.ZodType>(body: unknown, schema: S): z.output<S> {
  return schema.parse(body);
}

function zodParseSearxng<S extends z.ZodType>(body: unknown, schema: S): z.output<S> {
  return schema.parse(body);
}

function zodParseTavily<S extends z.ZodType>(body: unknown, schema: S): z.output<S> {
  return schema.parse(body);
}

function ollamaFetch(url: URL, payload: unknown): Promise<Response> {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

// サブワークフロー : ベクトル計算
class VectorCalculation {
  async run(baseUrl: string, arg: string): Promise<number[]> {
    const url = new URL("api/embed", baseUrl);
    const payload = this.getPayload(arg);
    const res = await ollamaFetch(url, payload);
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const body: unknown = await res.json();
    const data: CalculateVectorResult = zodParseEmbed(body, CalculateVectorSchema);
    return data.embeddings[0];
  }

  getPayload(arg: string): unknown {
    return {
      model: "bge-m3:567m",
      options: { num_ctx: 1024 },
      truncate: false,
      input: arg,
      keep_alive: "30m"
    };
  }
}

// サブワークフロー : 正規化
class Normalize {
  normalizeNfkc(original: string): string {
    return original.normalize('NFKC').toLowerCase().replace(/\s+/g, ' ').trim();
  }

  async normalizeSplitTokens(ollamaBaseUrl: string, nfkc: string): Promise<NormalizeSplitTokensResult> {
    const url = new URL("/api/chat", ollamaBaseUrl);
    const payload = this.normalizeSplitTokensPayload(nfkc);
    const res = await ollamaFetch(url, payload);
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const body: unknown = await res.json();
    const data: NormalizeSplitTokensResult = zodParseChat(body, NormalizeSplitTokensSchema);
    return data;
  }

  normalizeCrypto(keywords: string[]): NormalizeCryptoResult {
    keywords.sort();
    const data: NormalizeCryptoResult = {
      hashBase: keywords.join(' '),
      queryHash: createHash('sha256').update(keywords.join(' ')).digest('hex')
    };
    return data;
  }

  normalizeSplitTokensPayload(nfkc: string): unknown {
    return {
      model: "qwen3.5:2b",
      stream: false,
      think: false,
      format: {
        type: "object",
        properties: {
          keywords: {
            type: "array",
            items: { "type": "string" },
            maxItems: 5
          }
        },
        required: ["keywords"]
      },
      options: {
        temperature: 0,
        seed: 42,
        num_ctx: 1024,
        num_predict: 128
      },
      keep_alive: "30m",
      messages: [
        {
          role: "system",
          content: "ユーザーの発言から検索用キーワードを抽出し、JSONで返す。"
        },
        {
          role: "user",
          content: nfkc
        }
      ]
    };
  }
}

// サブワークフロー : キャッシュ・ベクトルストア検索
class SearchCacheAndVector {
  async basicCacheSearch(store: string, queryHash: string, keyVersion: string, ragdbResourceName: string): Promise<SearchResult> {
    const client = await this.getDbClient(ragdbResourceName);
    await client.connect();
    try {
      const result: pg.QueryResult<any> = await client.query(this.basicCacheSearchQuery(store), [queryHash, keyVersion]);
      console.log(result);
      const data: SearchResult = {
        source: "basicCacheSearch",
        hit: result.rowCount === null ? false : result.rowCount === 0 ? false : true,
        rows: result.rows
      };
      return data;
    }
    finally {
      await client.end();
    }
  }

  async vectorCacheSearch(store: string, vector: number[], keyVersion: string, ragdbResourceName: string): Promise<SearchResult> {
    const client = await this.getDbClient(ragdbResourceName);
    await client.connect();
    try {
      const result: pg.QueryResult<any> = await client.query(this.vectorCacheSearchQuery(store), [JSON.stringify(vector), keyVersion]);
      console.log(result);
      const data: SearchResult = {
        source: "vectorCacheSearch",
        hit: result.rowCount === null ? false : result.rowCount === 0 ? false : true,
        rows: result.rows
      };
      return data;
    }
    finally {
      await client.end();
    }
  }

  async vectorStoreSearch(store: string, vector: number[], ragdbResourceName: string): Promise<SearchResult> {
    const client = await this.getDbClient(ragdbResourceName);
    await client.connect();
    try {
      const result: pg.QueryResult<any> = await client.query(this.vectorStoreSearchQuery(store), [JSON.stringify(vector)]);
      console.log(result);
      const data: SearchResult = {
        source: "vectorStoreSearch",
        hit: result.rowCount === null ? false : result.rowCount === 0 ? false : true,
        rows: result.rows
      };
      return data;
    }
    finally {
      await client.end();
    }
  }

  async getDbClient(ragdbResourceName: string): Promise<pg.Client> {
    const db = await wmill.getResource(ragdbResourceName);
    return new pg.Client({
      host: db.host,
      port: db.port,
      user: db.user,
      password: db.password,
      database: db.dbname
    });
  }

  basicCacheSearchQuery(store: string): string {
    return `
      WITH 
      q AS ( 
      SELECT 
          results 
        , is_empty 
        , searched_at 
        , attempt_count 
        , COUNT(*) OVER() AS hit 
      FROM 
          ${store} 
      WHERE 
          cache_key = $1 
      AND search_method_version = $2 
      AND 
          ( 
          expires_at IS NULL 
       OR 
          expires_at > now() 
          ) 
      ) 
      SELECT 
          q.results 
        , q.is_empty 
        , q.searched_at 
        , q.attempt_count 
        , q.hit 
      FROM 
          q 
      WHERE 
          q.hit = ( 
              SELECT 
                  MAX(hit) 
              FROM 
                  q 
          ) 
      ;`;
  }

  vectorCacheSearchQuery(store: string): string {
    return `
      WITH 
      q AS ( 
      SELECT 
          results 
        , cache_key 
        , 1 - (query_embedding <=> $1::vector) AS similarity 
        , COUNT(*) OVER() AS hit 
      FROM 
          ( 
          SELECT 
              results 
            , cache_key 
            , query_embedding 
          FROM 
              ${store} 
          WHERE 
              query_embedding IS NOT NULL 
          AND search_method_version = $2 
          AND NOT is_empty 
          AND ( 
              expires_at IS NULL 
           OR 
              expires_at > now() 
              ) 
          ORDER BY 
              query_embedding <=> $1::vector 
          LIMIT 5 
          ) 
      ) 
      SELECT 
          q.results 
        , q.cache_key 
        , q.similarity 
      FROM 
          q 
      WHERE 
          q.hit = ( 
              SELECT 
                  MAX(hit) 
              FROM 
                  q 
              WHERE 
                  q.similarity >= 0.92 
          ) 
      ORDER BY 
          similarity DESC 
      ;`;
  }

  vectorStoreSearchQuery(store: string): string {
    return `
      WITH 
      q AS ( 
      SELECT 
          content 
        , metadata 
        , doc_id 
        , 1 - (embedding <=> $1::vector) AS similarity 
        , COUNT(*) OVER () AS hit 
      FROM 
          ( 
          SELECT 
              content 
            , metadata 
            , doc_id 
            , embedding 
          FROM 
              ${store} 
          ORDER BY 
              embedding <=> $1::vector 
          LIMIT 10 
          ) 
      ) 
      SELECT 
          q.content 
        , q.metadata 
        , q.doc_id 
      FROM 
          q 
      WHERE 
          q.hit = ( 
              SELECT 
                  MAX(hit) 
              FROM 
                  q 
              WHERE 
                  similarity >= 0.5 
          ) 
      ORDER BY 
          similarity DESC 
      ;`;
  }
}

// サブワークフロー : Web 検索
class SearchWeb {
  async searxngSearch(baseUrl: string, hashBase: string): Promise<SearxngResult> {
    const url = new URL("/searxng", baseUrl);
    url.searchParams.set("q", hashBase);
    url.searchParams.set("format", "json");
    url.searchParams.set("language", "ja");
    url.searchParams.set("categories", "general");
    url.searchParams.set("safesearch", "0");
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const body: unknown = await res.json();
    const data: SearxngResult = zodParseSearxng(body, SearxngSchema);
    return data;
  }

  async searxngCompletionJudge(baseUrl: string, nfkc: string, results: SearxngResult): Promise<SearxngCompletionJudgeResult> {
    const url = new URL("/api/chat", baseUrl);
    const payload = this.searxngCompletionJudgePayload(nfkc, results);
    const res = await ollamaFetch(url, payload);
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const body: unknown = await res.json();
    const data: SearxngCompletionJudgeResult = zodParseChat(body, SearxngCompletionJudgeSchema);
    return data;
  }

  async tavilySearch(baseUrl: string, hashBase: string, tavilyResourceName: string): Promise<TavilyResult> {
    const url = new URL("/search", baseUrl);
    const key = await wmill.getResource(tavilyResourceName);
    const payload = this.tavilySearchPayload(hashBase);
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${key.key}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const body: unknown = await res.json();
    const data: TavilyResult = zodParseTavily(body, TavilySchema);
    return data;
  }

  searxngCompletionJudgePayload(nfkc: string, results: SearxngResult): unknown {
    return {
      model: "qwen3.5:2b",
      stream: false,
      think: false,
      format: {
        type: "object",
        properties: {
          judge: { type: "boolean" }
        },
        required: ["judge"]
      },
      options: {
        temperature: 0,
        seed: 42,
        num_ctx: 8192,
        num_predict: 128
      },
      keep_alive: "30m",
      messages: [
        {
          role: "system",
          content: "<question> に答えるのに <result> が十分な情報を含むか判定する。関連し十分なら judge=true、無関係または不十分なら judge=false。"
        },
        {
          role: "user",
          content: JSON.stringify(`<question>${nfkc}</question><result>${JSON.stringify(results)}</result>`)
        }
      ]
    };
  }

  tavilySearchPayload(hashBase: string): unknown {
    return {
      query: hashBase,
      max_results: 5,
      search_depth: "basic",
      topic: "general",
      include_answer: false,
      include_raw_content: false,
      country: "japan"
    };
  }
}

// サブワークフロー : キャッシュ登録
class CacheRegister {
  async run(store: string, keyVersion: string, hashBase: string, queryHash: string, vector: number[], results: unknown, source: string, ragdbResourceName: string) {
    const client = await this.getDbClient(ragdbResourceName);
    await client.connect();
    try {
      const result: pg.QueryResult<any> = await client.query(this.registerQuery(store), [
        queryHash,
        keyVersion,
        hashBase,
        JSON.stringify(vector),
        JSON.stringify(results),
        source
      ]);
      console.log(result);
    }
    finally {
      await client.end();
    }
  }
  async getDbClient(ragdbResourceName: string): Promise<pg.Client> {
    const db = await wmill.getResource(ragdbResourceName);
    return new pg.Client({
      host: db.host,
      port: db.port,
      user: db.user,
      password: db.password,
      database: db.dbname
    });
  }

  registerQuery(store: string): string {
    return `
      INSERT INTO 
          ${store} 
          ( 
          cache_key 
        , search_method_version 
        , query_text 
        , query_embedding 
        , results 
        , source 
        , attempt_count 
        , searched_at 
        , expires_at 
          ) 
      VALUES 
          ( 
          $1 
        , $2 
        , $3 
        , $4::vector 
        , $5::jsonb 
        , $6 
        , 1 
        , now() 
        , now() 
          + 
          CASE 
          WHEN jsonb_array_length($5::jsonb) = 0 THEN 
              -- 空結果 = 短い負のキャッシュ
              interval '30 minutes' 
          ELSE 
              -- ヒット = 通常 TTL
              interval '7 days' 
          END 
          ) 
      ON 
      CONFLICT (cache_key, search_method_version) 
      DO 
      UPDATE 
      SET 
          query_text      = EXCLUDED.query_text 
        , query_embedding = COALESCE(EXCLUDED.query_embedding, ${store}.query_embedding) 
        , results         = EXCLUDED.results 
        , source          = EXCLUDED.source 
        , attempt_count   = ${store}.attempt_count + 1 
        , searched_at     = now() 
        , expires_at      = EXCLUDED.expires_at 
      RETURNING 
          cache_key 
        , is_empty 
        , attempt_count 
        , expires_at 
      ;`;
  }
}
