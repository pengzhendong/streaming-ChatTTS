import torch
from ChatTTS import model


class Embed(model.Embed):
    def __init__(
        self,
        hidden_size,
        num_audio_tokens,
        num_text_tokens,
        num_vq,
        model_path,
        device="cpu",
    ):
        super().__init__(hidden_size, num_audio_tokens, num_text_tokens, num_vq)
        self.from_pretrained(model_path, device=device)


class GPT(model.GPT):
    def __init__(
        self,
        config,
        embed_path,
        model_path,
        tokenizer_path,
        top_k=20,
        top_p=0.7,
        temperature=0.1,
        repetition_penalty=1.05,
        min_new_token=0,
        max_new_token=2048,
        manual_seed=None,
        compile=False,
        device="cpu",
    ):
        spk_stat = "愐穤巩噅廷戇笉屈癐媄垹垧帶爲漈塀殐慄亅倴庲舴猂瑈圐狴夥圓帍戛挠腉耐劤坽喳幾战謇聀崒栄呥倸庭燡欈杁襐褄乭埗幺爃弔摁斐捔兕佖廐舏竾豃磐姓趡佄幒爚欄豄讐皳訵仩帆投謌荃蝐叄圝伆幦抂茁呄掑斃讹傮庞爣蜀橁偐祄亥兡常爂欍扉丐浔佱僈強払伅扂蛐徴憍傞巀戺欀艂琐嗴啥値彷刂權穈扒卤俔贲庛初笂卄贐枴仭亁庛剎猢扃缐趤刁偵幪舏伌煁婐潤晍位弾舙茥穁葏蠣訑企庤刊笍橁溑僔云偁庯戚伍潉膐脴僵噔廃艅匊祂唐憴壝嗙席爥欁虁谐牴帽势弿牳蜁兀蛐傄喩丿帔刔圆衁廐罤庁促帙劢伈汄樐檄勵伴弝舑欍罅虐昴劭勅帜刼朊蕁虐蓴樑伫幨扑謪剀堐稴丵伱弐舮諸赁習俔容厱幫牶謃孄糐答嗝僊帜燲笄終瀒判久僤帘爴茇千孑冄凕佳引扐蜁歁缏裄剽儺恘爋朏眿廐呄塍嘇幻爱茠詁訐剴唭俐幾戊欀硁菐贄楕偒巡爀弎屄莐睳賙凶彎刅漄區唐溴剑劋庽舽猄煃跐夔惥伾庮舎伈罁垑坄怅业怯刁朇獁嶏覔坩俳巶爜朐潁崐萄俹凛常爺笌穀聐此夡倛帡刀匉終窏舣販侽怿扉伥贿憐忓謩姆幌犊漂慆癒却甝兎帼戏欅詂浐朔仹壭帰臷弎恇菐獤帡偖帘爞伅腂皐纤囅充幓戠伥灂丐訤戱倱弋爮嬌癁恐孄侥劬忶刓國詀桒古偩嘄庬戚茝赂监燤嘑勌幦舽持呂諐棤姑再底舡笍艃瀐孴倉傔弋爔猠乁濑塄偽嘧恂舛缇襃厐窴仡刱忕別漇穁岏缴廽价庌爊謈硄讑惤倁儂庭爋伇蝂嶐莔摝傠库刞茄歃戏薤伍伯廮创笠塄熐兴勽俄帅剉最腀砐敤卝侍弆戺朒虃旐蚄梕亖幔牻朣扅贐玔堝噅帡剌圅摀崐彤流僳庙爖嬇啁渐悤堁丛幆刧挜彃悐幤刹嚟恕芁看聀摐焔向乁帖爭欁癃糒圄弙佱廜戤謍婀咐昴焍亩廦艏拼謿芐癤怹兽幸舳朇畁喐稔毝丼弈懲挀譂勑哴啁伎常舭笯晁堑俄叩剔廟爍欦絁夒伤休傑廳戌蜅潆癐彴摑勯床刽欅艁砐忄搉从廡舊猥潂唐委仱僜廼爤朄呃弐礔滵垓幩爄挂筁乐籤刕凟幵爠弉癅乑吴勥伖帪舩茆婁碐幤叭乢巜艳猁桀桐啄唩俊幍舮猀艅焐螔琽亀帋爜缅噃咐斤喩予幩爛笆摀浐猴依侹幃刕園慄蛐栤澹仑座爼謉桃慐浔斕偻幛懰嬓衁愐氄悅仿应芔漄衃敐謤傁匩幹抃圉癄廐裄屵噉幍利謍聂搐蛔嚙坍怗舁圐畃膐栄刵东巆戤諾呃偑媤嗨跞忶爝眄祂朒嶔僭劉忾刐匋癄袐翴珅僷廲芄茈恈皐擄崑伄廉牍匃剃犏澤唑丄庺戃伃煀某杄偙亽帴切缌罄挐尴噙倰带舞漄橄塐糴俩僯帀般漀坂栐更両俇廱舌猁慂拐偤嶱卶应刪眉獁茐伔嘅偺帟舊漂恀栐暄喡乞庙舆匂敀潑恔劑侖延戦盽怶唯慳蝘蟃孫娎益袰玍屃痶翮笪儚裀倹椌玻翀詵筽舘惯堿某侰晈藏缮詗廦夸妎瑻瀒裔媀憞唃冶璭狻渠荑奬熹茅愺氰菣滠翦岓褌泣崲嚭欓湒聙宺爄蛅愸庍匃帆誔穮懌蓪玷澌氋抌訙屌臞廛玸听屺希疭孝凂紋新煎彃膲跱尪懁眆窴珏卓揨菸紭概囥显壌榄垫嘮嬭覤媸侵佮烒耸觌婀秋狃帹葯訤桜糨笾腢伀肶悍炂艤禖岅臺惘梷瞍友盁佨岧憳瓧嘴汬藊愌蘤嶠硴绤蜲襏括勾谂縨妥蓪澭竭萢藜纞糲煮愆瀯孯琓罂諺塿燗狟弙衯揻縷丱糅臄梱瀮杰巳猙亊符胠匃泀廏圃膂蒃籏礩岈簹缌劺燲褡孓膜拔蠿觮呋煣厌尷熜論弲牭紫寊誃紀橴賬傸箍弚窃侫簲慯烣渽祌壓媥噜夽夛諛玹疮禄冪謇媽衤盰缺繑薫兾萧嵱打滽箺嚯凣狢蠜崼覽烸簶盯籓摀苶峸懗泲涻凮愳緗剋笔懆廡瞿椏礤惐藥崍腈烄伹亯昣翬褍絋桫僨吨莌丛矄蜞娈憊苆塁蓏嚢嫼绻崱婋囱蠸篯晣芀繼索兓僖誹岯圪褰蠇唓妷胅巁渮砛傈蝷嵚冃購赁峍裋荂舾符熻岳墩寮粃凲袑彚太绲头摯繳狁俥籌冝諝註坎幫擤詒宒凕賐唶梎噔弼課屿覍囨焬櫱撪蝮蝬簸懰櫫涺嵍睻屪翔峞慘滟熲昱军烊舿尦舄糖奁溏凂彆蝲糴禍困皻灏牋睒诙嶱臀开蓈眎腼丢纻廏憤嫖暭袭崲肸螛妒榗紉谨窮袃瑠聍绊腆亿冲葐喋縔詖岑兾给堸赏旻桀蛨媆訂峦紷敯囬偐筨岸焸拭笵殒哜墒萍屓娓諙械臮望摰芑寭准僞谹氍旋憢菮屃划欣瘫谎蘻哐繁籥禦僿誵皯墓燀縿笞熦绗稹榎矻綞蓓帡戓沺区才畃洊詪糐裶盰窶耎偌劂誐庩惝滜沺哮呃煐譠崄槀猄肼蔐擋湌蠺篃恥諌瞦宍堫挪裕崑慩狲悠煋仛愞砈粵八棁害楐妋萔貨尵奂苰怫誎傫岆蕯屇脉夈仆茎刓繸芺壸碗曛汁戭炻獻凉媁兎狜爴怰賃纎袏娷禃蓥膹薪渻罸窿粫凾褄舺窮墫干苊繁冏僮訸夯绛蓪虛羽慲烏憷趎睊蠰莍塞成廎盁欏喓蜮譤崆楁囘矇薭伣艘虝帴奮苢渶虎暣翐蝃尾稈糶瀴罐嵚氮葯笫慐棌悶炯竻爅们媡姢嫺窷刮歫劈裩屬椕賑蜹薊刲義哯尗褦瓀稾礋揣窼舫尋姁椄侸嗫珺修纘媃腽蛛稹梭呛瀈蘟縀礉論夵售主梮蠉娅娭裀誼嶭観枳倊簈褃擞綿催瞃溶苊笛襹櫲盅六囫獩佃粨慯瓢眸旱荃婨蔞岋祗墼焻网牻琖詆峋秉胳媴袭澓賢経稟壩胫碯偏囫嶎纆窈槊賐撹璬莃缘誾宭愊眗喷监劋萘訯總槿棭戾墮犄恌縈簍樥蛔杁袭嫛憫倆篏墵賈羯茎觳蒜致娢慄勒覸蘍曲栂葭宆妋皽缽免盳猼蔂糥觧烳檸佯憓煶蔐筼种繷琲膌塄剰讎対腕棥渽忲俛浪譬秛惛壒嘸淫冻曄睻砃奫貯庴爅粓脮脡娎妖峵蘲討惋泊蠀㴆"
        embed = Embed(
            config["hidden_size"],
            config["num_audio_tokens"],
            config["num_text_tokens"],
            config["num_vq"],
            embed_path,
            device,
        )
        super().__init__(
            gpt_config=config,
            embed=embed,
            use_flash_attn=False,
            use_vllm=False,
            device=device,
            device_gpt=device,
        )
        self.eval()
        self.from_pretrained(model_path, embed_path)
        self.prepare(compile=True and "cuda" in str(device))

        self.config = config
        self.device = device
        self.embed = embed
        self.speaker = model.Speaker(config["hidden_size"], spk_stat, device)
        self.tokenizer = model.Tokenizer(tokenizer_path)

        self.num_code = config["num_audio_tokens"] - 1
        logits_warpers, logits_processors = model.gen_logits(
            num_code=self.num_code,
            top_P=top_p,
            top_K=top_k,
            repetition_penalty=repetition_penalty,
        )
        self.logits_processors = (*logits_processors, *logits_warpers)
        self.temperature = torch.tensor([temperature] * config["num_vq"], device=device)
        self.max_new_token = max_new_token
        self.min_new_token = min_new_token
        self.manual_seed = manual_seed
        self.random_spk_emb = self.speaker.sample_random()

    def generate(self, text, spk_emb=None, speech_tokens=None, transcript=""):
        transcript = f"{transcript}[speed_5]"
        spk_emb = spk_emb or self.random_spk_emb
        text = self.speaker.decorate_code_prompts([text], transcript, None, spk_emb)
        # speech_tokens: dvae.encode(wav)
        if speech_tokens is not None:
            speech_tokens = self.speaker.decode_prompt(speech_tokens)
        input_ids, attention_mask, text_mask = self.tokenizer.encode(
            text, self.config["num_vq"], speech_tokens, self.device
        )
        emb = self.embed(input_ids, text_mask)
        if spk_emb is not None:
            self.speaker.apply(
                emb,
                spk_emb,
                input_ids,
                self.tokenizer.spk_emb_ids,
                self.device,
            )

        num_tokens = 0
        for tokens in super().generate(
            emb,
            input_ids,
            temperature=self.temperature,
            eos_token=self.num_code,
            attention_mask=attention_mask,
            max_new_token=self.max_new_token,
            min_new_token=self.min_new_token,
            logits_processors=self.logits_processors,
            infer_text=False,
            return_hidden=True,
            stream=True,
            show_tqdm=True,
            ensure_non_empty=True,
            stream_batch=4,
            manual_seed=self.manual_seed,
        ):
            ids = tokens.ids[0].T[None, :, num_tokens:]
            num_tokens += ids.shape[2]
            yield ids

