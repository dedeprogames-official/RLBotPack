from tools import *
from objects import *
from routines import *

import math


# ============================================================
# Helpers de desenho (render) - compatível mesmo sem "circle"
# ============================================================

def _safe_line(agent, a, b, color=(255, 255, 255)):
    try:
        agent.line(a, b, color)
        return True
    except Exception:
        # se sua versão usa renderer direto, você pode adaptar aqui
        return False


def _draw_circle(agent, center, radius=250, color=(255, 255, 255), steps=24, z_override=None):
    # desenha círculo aproximado com linhas
    pts = []
    z = center.z if z_override is None else z_override
    for i in range(steps + 1):
        t = (i / steps) * (math.pi * 2)
        pts.append(Vector3(center.x + math.cos(t) * radius, center.y + math.sin(t) * radius, z))

    for i in range(len(pts) - 1):
        _safe_line(agent, pts[i], pts[i + 1], color)


def _draw_arrow(agent, start, end, color=(255, 255, 255), head_len=180, head_angle_deg=28):
    _safe_line(agent, start, end, color)

    # cabeça da seta no "end"
    dir_vec = (end - start)
    if dir_vec.magnitude() < 1:
        return
    d = dir_vec.normalize()

    # cria 2 vetores laterais no plano XY
    ang = math.radians(head_angle_deg)
    # rotaciona d no plano XY
    left = Vector3(
        d.x * math.cos(ang) - d.y * math.sin(ang),
        d.x * math.sin(ang) + d.y * math.cos(ang),
        0
    )
    right = Vector3(
        d.x * math.cos(-ang) - d.y * math.sin(-ang),
        d.x * math.sin(-ang) + d.y * math.cos(-ang),
        0
    )

    p1 = end - left.normalize() * head_len
    p2 = end - right.normalize() * head_len
    _safe_line(agent, end, p1, color)
    _safe_line(agent, end, p2, color)


def _draw_text_3d(agent, location, text, color=(255, 255, 255)):
    """
    GoslingUtils normalmente tem renderer interno, mas algumas versões expõem helpers.
    Tentamos várias opções sem quebrar.
    """
    # tenta métodos comuns
    for attr in ["draw_string_3d", "string", "text", "draw_text_3d"]:
        fn = getattr(agent, attr, None)
        if callable(fn):
            try:
                fn(location, text, color)
                return True
            except Exception:
                pass

    # tenta renderer
    renderer = getattr(agent, "renderer", None)
    if renderer is not None:
        for attr in ["draw_string_3d", "draw_string_2d", "draw_string"]:
            fn = getattr(renderer, attr, None)
            if callable(fn):
                try:
                    # muitas versões: draw_string_3d(x,y,z, scaleX, scaleY, text, color)
                    # outras: draw_string_3d(vec, scale, text, color)
                    # vamos tentar formatos
                    try:
                        fn(location, 1, 1, text, color)
                        return True
                    except Exception:
                        try:
                            fn(location.x, location.y, location.z, 1, 1, text, color)
                            return True
                        except Exception:
                            pass
                except Exception:
                    pass

    return False


# ============================================================
# Helpers de "previsão" simples da bola (para render e timing)
# ============================================================

GRAVITY_Z = -650.0  # Rocket League approx

def _ballistic_ball_pos(ball_loc, ball_vel, dt):
    # movimento simples sem bounce (serve p/ debug render)
    return Vector3(
        ball_loc.x + ball_vel.x * dt,
        ball_loc.y + ball_vel.y * dt,
        ball_loc.z + ball_vel.z * dt + 0.5 * GRAVITY_Z * dt * dt
    )


def _ballistic_ball_vel(ball_vel, dt):
    return Vector3(ball_vel.x, ball_vel.y, ball_vel.z + GRAVITY_Z * dt)


def _sample_ball_path(ball_loc, ball_vel, t_end, step=0.10):
    pts = []
    t = 0.0
    loc = ball_loc
    vel = ball_vel
    while t < t_end:
        pts.append(loc)
        # integra
        loc = _ballistic_ball_pos(loc, vel, step)
        vel = _ballistic_ball_vel(vel, step)

        # clamp chão (sem bounce real, mas evita "z negativa" no debug)
        if loc.z < 0:
            loc = Vector3(loc.x, loc.y, 0)
            vel = Vector3(vel.x, vel.y, 0)

        t += step
    pts.append(loc)
    return pts


# ============================================================
# Helpers de estratégia / matemática
# ============================================================

def _cap(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _eta_to_point(agent, car, point):
    """
    ETA simples: distância / velocidade + penalidade por ângulo.
    Bom o suficiente pra rotação/commit.
    """
    to = point - car.location
    dist = to.magnitude()
    if dist < 1:
        return 0.0

    local = car.local(to)
    angle = abs(math.atan2(local.x, local.y))  # 0 -> alinhado, pi -> de costas

    speed = max(300.0, car.velocity.magnitude())
    base = dist / speed

    # penaliza giro
    turn_penalty = (angle / math.pi) * 1.2  # até +1.2s
    # penaliza se estiver quase parado
    slow_penalty = 0.3 if speed < 900 else 0.0

    return base + turn_penalty + slow_penalty


def _role_and_ranks(agent):
    """
    Retorna:
      - my_role: "FIRST" / "SECOND" / "THIRD"
      - i_am_last_man: bool (mais perto do nosso gol)
      - etas: lista (player_index, eta)
      - goal_dists: lista (player_index, dist_own_goal)
    """
    friends = list(getattr(agent, "friends", []))
    players = [agent.me] + friends
    ball_loc = agent.ball.location
    own_goal = agent.friend_goal.location

    # 1v1: não existe rotação 1/2/3 man; evitar comportamento "chuta e volta pro gol" sempre
    if len(players) == 1:
        eta = _eta_to_point(agent, agent.me, ball_loc)
        gd = (agent.me.location - own_goal).magnitude()
        return "FIRST", False, [(agent.me.index, eta)], [(agent.me.index, gd)]

    etas = []
    goal_dists = []

    for p in players:
        etas.append((p.index, _eta_to_point(agent, p, ball_loc)))
        goal_dists.append((p.index, (p.location - own_goal).magnitude()))

    etas_sorted = sorted(etas, key=lambda x: x[1])
    goal_sorted = sorted(goal_dists, key=lambda x: x[1])

    # rank do meu index na lista de ETA
    my_idx = agent.me.index
    my_rank_eta = [i for i, (idx, _) in enumerate(etas_sorted) if idx == my_idx][0]
    my_rank_goal = [i for i, (idx, _) in enumerate(goal_sorted) if idx == my_idx][0]

    # regra simples p/ role:
    # - FIRST: menor ETA
    # - THIRD: mais perto do próprio gol (last man)
    # - SECOND: resto
    i_am_last_man = (my_rank_goal == 0)

    if my_rank_eta == 0 and not i_am_last_man:
        my_role = "FIRST"
    elif i_am_last_man:
        my_role = "THIRD"
    else:
        my_role = "SECOND"

    return my_role, i_am_last_man, etas_sorted, goal_sorted


def _threat_level(agent):
    """
    Heurística de perigo:
    - bola indo pro nosso gol
    - bola no nosso lado
    - oponente com ETA menor que o nosso
    """
    s = side(agent.team)

    ball = agent.ball
    me = agent.me
    own_goal = agent.friend_goal.location

    ball_on_our_side = (ball.location.y * s) < 0
    ball_towards_our_goal = (ball.velocity.y * s) < -200
    ball_close_to_goal = (ball.location - own_goal).magnitude() < 2800

    # melhor ETA do inimigo vs nosso
    foe_best_eta = 999
    for f in getattr(agent, "foes", []):
        foe_best_eta = min(foe_best_eta, _eta_to_point(agent, f, ball.location))
    my_eta = _eta_to_point(agent, me, ball.location)

    foe_beats_me = foe_best_eta + 0.10 < my_eta

    threat = 0.0
    if ball_on_our_side: threat += 0.35
    if ball_towards_our_goal: threat += 0.35
    if ball_close_to_goal: threat += 0.35
    if foe_beats_me: threat += 0.35

    return _cap(threat, 0.0, 1.5)


def _desired_approach_speed(agent, dist_to_ball, intercept_z, ball_vz):
    """
    Controle inteligente (4):
    - Se intercepto é "baixo" → pode chegar mais rápido
    - Se bola tá caindo ou vai quicar → reduz pra chegar na hora do toque
    """
    # base por distância
    if dist_to_ball > 2200:
        base = 2300
    elif dist_to_ball > 1400:
        base = 2000
    elif dist_to_ball > 800:
        base = 1600
    else:
        base = 1100

    # bola no ar / caindo: desacelera mais pra timing
    if intercept_z > 160:
        base -= 400
    if ball_vz < -200 and intercept_z > 120:
        base -= 300

    return _cap(base, 600, 2300)


def _choose_shot_target(agent, shot_kind):
    # pra render/seta: um ponto “pra onde” queremos chutar
    if shot_kind == "goal":
        return agent.foe_goal.location
    if shot_kind == "clear":
        # limpa pro lado oposto do centro (evita devolver pro meio)
        s = side(agent.team)
        x = 3800 if agent.ball.location.x < 0 else -3800
        y = 1800 * s  # empurrando pra frente
        return Vector3(x, y, 0)
    return agent.foe_goal.location


def _shot_score(agent, shot, shot_kind="goal"):
    """
    (3) Scoring melhor:
    - velocidade média até o intercepto
    - alinhamento car->bola e bola->alvo
    - penaliza altura se não tiver boost
    - penaliza se for overcommit (threat alto)
    """
    me = agent.me
    ball = agent.ball
    now = agent.time

    # intercept_time / ball_location existem nos shots do GoslingUtils normalmente
    intercept_time = getattr(shot, "intercept_time", None)
    ball_loc = getattr(shot, "ball_location", None)

    if intercept_time is None or ball_loc is None:
        return -999999

    dt = max(0.01, intercept_time - now)
    dist = (ball_loc - me.location).magnitude()
    avg_speed = dist / dt

    # alinhamentos
    to_ball = (ball_loc - me.location)
    to_ball_n = to_ball.normalize() if to_ball.magnitude() > 1 else Vector3(0, 1, 0)

    target_point = _choose_shot_target(agent, shot_kind)
    ball_to_target = (target_point - ball_loc)
    ball_to_target_n = ball_to_target.normalize() if ball_to_target.magnitude() > 1 else Vector3(0, 1, 0)

    # quanto o carro está “apontado” pro intercepto (aprox por local.y)
    local = me.local(to_ball)
    facing = _cap(local.y / (to_ball.magnitude() + 1e-6), -1, 1)  # ~cos
    align = _cap(to_ball_n.dot(ball_to_target_n), -1, 1)

    # ratio do find_hits (quando existir)
    ratio = getattr(shot, "ratio", 1.0)

    # boost / altura
    height_penalty = 0.0
    if ball_loc.z > 220 and me.boost < 40:
        height_penalty += 0.7
    if ball_loc.z > 380 and me.boost < 60:
        height_penalty += 1.2

    # risco (se jogo perigoso, evita commits ruins)
    threat = _threat_level(agent)
    risk_penalty = 0.0
    if threat > 0.9 and shot_kind == "goal":
        risk_penalty += 0.6

    # score final
    score = (avg_speed * 0.55) + (ratio * 900) + (align * 500) + (facing * 300)
    score -= (height_penalty * 900)
    score -= (risk_penalty * 800)

    return score


def _pick_best_shot(agent, shots_dict):
    best = None
    best_kind = None
    best_score = -999999

    for kind in ["goal", "clear"]:
        for shot in shots_dict.get(kind, []):
            sc = _shot_score(agent, shot, kind)
            if sc > best_score:
                best = shot
                best_score = sc
                best_kind = kind

    return best, best_kind, best_score


def _shot_is_aerial(shot):
    name = shot.__class__.__name__.lower()
    return ("aerial" in name) or ("air" in name)


def _has_routine(name):
    return name in globals() and callable(globals()[name])


# ============================================================
# (NOVO) AERIAL: escolher ponto futuro inteligente + tempo de chegar
# ============================================================

def _aerial_time_needed(agent, intercept_loc):
    """
    Estimativa simples do tempo mínimo pro aerial acontecer com chance:
    - inclui "setup" (alinhar + pular) e viagem horizontal/vertical
    - escala por boost baixo (mais lento/menos controle)
    """
    me = agent.me

    dx = intercept_loc.x - me.location.x
    dy = intercept_loc.y - me.location.y
    d_xy = math.sqrt(dx * dx + dy * dy)

    h = max(0.0, intercept_loc.z - me.location.z)

    setup = 0.55
    travel = d_xy / 1700.0
    climb = max(0.0, h - 140.0) / 900.0
    margin = 0.10

    t = setup + travel + climb + margin

    # boost baixo => precisa de mais tempo (aéreo mais lento e menos correção)
    if me.boost < 55:
        t *= 1.10
    if me.boost < 35:
        t *= 1.18
    if me.boost < 20:
        t *= 1.28

    return t


def _pick_best_aerial(agent, shots_dict):
    """
    Escolhe um shot AÉREO que:
    - esteja no ar (z alto)
    - seja futuro o suficiente para dar tempo de chegar
    - maximize o score existente + bônus por 'margem de tempo'
    """
    now = agent.time
    me = agent.me

    best = None
    best_kind = None
    best_score = -999999

    for kind in ["goal", "clear"]:
        for shot in shots_dict.get(kind, []):
            it = getattr(shot, "intercept_time", None)
            il = getattr(shot, "ball_location", None)
            if it is None or il is None:
                continue

            # precisa estar no ar pra valer como aerial
            if il.z < 260:
                continue

            dt = it - now
            if dt <= 0.0:
                continue

            need = _aerial_time_needed(agent, il)

            # precisa ser um ponto futuro que dá tempo de chegar
            if dt < (need + 0.08):
                continue

            # muito longe no futuro costuma gerar aerial ruim/aleatório
            if dt > 4.0:
                continue

            # score base do shot + bônus por ter folga de tempo (mas sem exagero)
            base_sc = _shot_score(agent, shot, kind)
            slack = _cap(dt - need, 0.0, 1.25)
            sc = base_sc + slack * 220.0

            # se boost muito baixo, exige ainda mais folga prática
            if me.boost < 25 and slack < 0.35:
                sc -= 600.0

            if sc > best_score:
                best = shot
                best_score = sc
                best_kind = kind

    return best, best_kind, best_score


# ============================================================
# Rotinas custom (kickoff cheat) – sem depender de rotinas do lib
# ============================================================

class CheatKickoff:
    """
    Vai “cheatar” no kickoff: avança até um ponto seguro e pronto pra pegar rebote.
    """
    def __init__(self, spot):
        self.spot = spot

    def run(self, agent):
        relative = self.spot - agent.me.location
        dist = relative.magnitude()
        local = agent.me.local(relative)

        defaultPD(agent, local)
        defaultThrottle(agent, _cap(dist * 2, 0, 2300))
        agent.controller.boost = (dist > 1400 and abs(local.x) < 250 and abs(local.y) > 800)

        # termina quando chega
        if dist < 250:
            return True  # pop
        return False


# ============================================================
# (NOVO) Wavedash (wayyshot) + render
# ============================================================

class WaveDash:
    """
    Wavedash simples por temporização + detecção de descida.
    - 1) jump curto
    - 2) espera cair
    - 3) dodge pra frente quando perto do chão
    """
    def __init__(self):
        self.t0 = None
        self.phase = 0
        self.dodge_t = None

    def run(self, agent):
        me = agent.me

        # se já não está no chão ao iniciar, não força
        if self.t0 is None:
            self.t0 = agent.time
            self.phase = 0
            self.dodge_t = None
            if not bool(getattr(me, "on_ground", True)):
                return True

        elapsed = agent.time - self.t0

        # render
        _draw_text_3d(agent, me.location + Vector3(0, 0, 140), "WAVEDASH", (255, 120, 255))

        # defaults
        agent.controller.throttle = 1.0
        agent.controller.boost = False
        agent.controller.handbrake = False
        agent.controller.steer = 0.0
        agent.controller.yaw = 0.0
        agent.controller.roll = 0.0

        # fase 0: jump curto
        if self.phase == 0:
            agent.controller.jump = True
            agent.controller.pitch = -1.0
            if elapsed > 0.08:
                self.phase = 1
            return False

        # fase 1: solta jump e inclina pra frente
        if self.phase == 1:
            agent.controller.jump = False
            agent.controller.pitch = -1.0
            if elapsed > 0.22:
                self.phase = 2
            return False

        # fase 2: espera cair e tocar perto do chão
        if self.phase == 2:
            agent.controller.jump = False
            agent.controller.pitch = -1.0

            # quando estiver descendo e perto do chão, executa dodge
            if me.location.z < 28 and me.velocity.z < -50:
                self.phase = 3
                self.dodge_t = agent.time
            # timeout (se algo der errado)
            if elapsed > 1.20:
                return True
            return False

        # fase 3: dodge curto pra frente
        if self.phase == 3:
            if self.dodge_t is None:
                self.dodge_t = agent.time

            dt = agent.time - self.dodge_t
            agent.controller.pitch = -1.0
            agent.controller.jump = (dt < 0.06)

            if dt > 0.14:
                return True
            return False

        return True


# ============================================================
# BOT
# ============================================================

class ExampleBot(GoslingAgent):
    def run(agent):
        # ======================
        # State inicial
        # ======================
        if not hasattr(agent, "dbg"):
            agent.dbg = {
                "action": "INIT",
                "role": "UNK",
                "intercept_t": None,
                "intercept_loc": None,
                "shot_kind": None,
                "shot_target": None
            }

        # wavedash cooldown
        if not hasattr(agent, "wd_last"):
            agent.wd_last = -9999.0

        me = agent.me
        ball = agent.ball
        s = side(agent.team)

        # ======================
        # Role / rotação
        # ======================
        my_role, i_am_last_man, etas_sorted, goal_sorted = _role_and_ranks(agent)
        threat = _threat_level(agent)

        # texto da ação no carro (vai atualizando)
        agent.dbg["role"] = my_role

        # ======================
        # RENDER base: linhas de debug do seu exemplo
        # ======================
        try:
            left_test_a = Vector3(-4100 * s, ball.location.y, 0)
            left_test_b = Vector3(4100 * s, ball.location.y, 0)
            _safe_line(agent, me.location, left_test_a, (0, 255, 0))
            _safe_line(agent, me.location, left_test_b, (255, 0, 0))
        except Exception:
            pass

        # ======================
        # Se já tem rotina rodando, só render e sai
        # ======================
        if len(agent.stack) > 0:
            # render texto
            _draw_text_3d(agent, me.location + Vector3(0, 0, 120),
                          f"[{agent.dbg['role']}] {agent.dbg['action']}", (255, 255, 255))
            return

        # ======================
        # KICKOFF avançado (8)
        # ======================
        if agent.kickoff_flag:
            # quem é o taker? menor ETA da equipe
            my_is_taker = (etas_sorted[0][0] == me.index)

            if my_is_taker:
                agent.dbg["action"] = "KICKOFF: TAKING"
                if _has_routine("kickoff"):
                    agent.push(kickoff())
                else:
                    # fallback: acelera reto pra bola
                    agent.controller.throttle = 1
                    agent.controller.boost = True
            else:
                # cheat spot: ligeiramente à frente do meio, do nosso lado
                cheat_spot = Vector3(0, -800 * s, 0)
                agent.dbg["action"] = "KICKOFF: CHEAT"
                agent.push(CheatKickoff(cheat_spot))

            _draw_text_3d(agent, me.location + Vector3(0, 0, 120),
                          f"[{agent.dbg['role']}] {agent.dbg['action']}", (255, 255, 255))
            return

        # ======================
        # Targets e shots (3) + Aerials (6)
        # ======================
        targets = {
            "goal": (agent.foe_goal.left_post, agent.foe_goal.right_post),
            "clear": (Vector3(-4100 * s, ball.location.y, 0), Vector3(4100 * s, ball.location.y, 0)),
        }

        shots = find_hits(agent, targets)

        best_shot, best_kind, best_score = _pick_best_shot(agent, shots)

        # ======================
        # Intercept info p/ render (círculo / trajeto / seta)
        # ======================
        intercept_time = None
        intercept_loc = None
        if best_shot is not None:
            intercept_time = getattr(best_shot, "intercept_time", None)
            intercept_loc = getattr(best_shot, "ball_location", None)

        agent.dbg["intercept_t"] = intercept_time
        agent.dbg["intercept_loc"] = intercept_loc
        agent.dbg["shot_kind"] = best_kind
        agent.dbg["shot_target"] = _choose_shot_target(agent, best_kind) if best_kind else None

        # ======================
        # DECISÃO PRINCIPAL
        # ======================

        # Regra de commit por rotação:
        # - THIRD (last man) só comita em clear/save ou se ameaça baixa
        can_commit_attack = (my_role != "THIRD") or (threat < 0.55)

        # Se existe shot bom
        if best_shot is not None:
            is_aerial = _shot_is_aerial(best_shot)
            z = intercept_loc.z if intercept_loc else 0

            # (6) Aerial: agora escolhe ponto FUTURO inteligente que dá tempo de chegar
            if is_aerial or z > 250:
                aerial_shot, aerial_kind, aerial_sc = _pick_best_aerial(agent, shots)

                if aerial_shot is not None and me.boost >= 35 and can_commit_attack:
                    # atualiza debug/intercept para render apontar pro ponto certo do aerial
                    a_it = getattr(aerial_shot, "intercept_time", None)
                    a_il = getattr(aerial_shot, "ball_location", None)

                    agent.dbg["intercept_t"] = a_it
                    agent.dbg["intercept_loc"] = a_il
                    agent.dbg["shot_kind"] = aerial_kind
                    agent.dbg["shot_target"] = _choose_shot_target(agent, aerial_kind) if aerial_kind else None

                    agent.dbg["action"] = f"SHOT: AERIAL ({aerial_kind})"
                    agent.push(aerial_shot)
                else:
                    agent.dbg["action"] = "HOLD: NO GOOD AERIAL POINT / SAFE ROTATION"

            else:
                # (4) Speed control na aproximação
                # Se o shot for cedo demais / perto demais, às vezes a gente chega “antes”.
                # Então só comita se score ok e role permite.
                if best_score > 500 and can_commit_attack:
                    agent.dbg["action"] = f"SHOT: GROUND ({best_kind})"
                    agent.push(best_shot)
                else:
                    agent.dbg["action"] = "POSITION: WAIT / SUPPORT"

        # Sem shot: defesa / rotação / boost
        if len(agent.stack) < 1:
            # Perigo alto => defender
            if threat > 0.85 or my_role == "THIRD":
                # tenta usar save do GoslingUtils se existir e bola indo pro gol
                ball_towards_our_goal = (ball.velocity.y * s) < -150
                ball_close_goal = (ball.location - agent.friend_goal.location).magnitude() < 3200

                if ball_towards_our_goal and ball_close_goal and _has_routine("save"):
                    agent.dbg["action"] = "DEFENSE: SAVE ROUTINE"
                    agent.push(save(agent.friend_goal.location))
                else:
                    # fallback: ir pro far post com velocidade controlada
                    left_dist = (agent.friend_goal.left_post - me.location).magnitude()
                    right_dist = (agent.friend_goal.right_post - me.location).magnitude()
                    target = agent.friend_goal.left_post if left_dist < right_dist else agent.friend_goal.right_post

                    # move um pouco pra fora do gol (far post + offset)
                    target = Vector3(target.x, target.y, 0)

                    agent.dbg["action"] = "DEFENSE: FAR POST / SHADOW"

                    relative = target - me.location
                    dist = relative.magnitude()
                    local = me.local(relative)

                    # (NOVO) wavedash quando sem boost e longe, pra acelerar sem gastar boost
                    if (me.boost < 12 and dist > 2600 and me.velocity.magnitude() < 1050 and
                        abs(local.x) < 220 and (agent.time - agent.wd_last) > 2.2 and
                        bool(getattr(me, "on_ground", True))):
                        agent.wd_last = agent.time
                        agent.dbg["action"] = "MOVE: WAVEDASH"
                        agent.push(WaveDash())
                    else:
                        defaultPD(agent, local)

                        # velocidade segura (não torrar boost)
                        speed = _cap(dist * 1.8, 0, 2000)
                        defaultThrottle(agent, speed)
                        agent.controller.boost = (dist > 2600 and abs(local.x) < 200 and abs(local.y) > 900 and me.boost > 30)

            # Boost se estiver baixo e não for last man
            elif me.boost < 30:
                best_boost = None
                best_val = -1.0
                for boost in agent.boosts:
                    if not boost.active:
                        continue
                    if not boost.large:
                        continue

                    me_to_boost = (boost.location - me.location).normalize()
                    boost_to_goal = (agent.friend_goal.location - boost.location).normalize()

                    val = boost_to_goal.dot(me_to_boost)
                    if val > best_val:
                        best_val = val
                        best_boost = boost

                if best_boost is not None and _has_routine("goto_boost"):
                    agent.dbg["action"] = "RESOURCE: GET BOOST (LARGE)"
                    agent.push(goto_boost(best_boost, agent.friend_goal.location))

            # Support: posicionar mid/back post “inteligente”
            else:
                agent.dbg["action"] = "ROTATE: SUPPORT MID"
                # spot de suporte: entre bola e nosso gol, um pouco pra trás
                goal = agent.friend_goal.location
                ball_to_goal = (goal - ball.location).normalize()
                support = ball.location + ball_to_goal * 2200  # 2200 atrás da bola
                support = Vector3(_cap(support.x, -3800, 3800), _cap(support.y, -5100, 5100), 0)

                relative = support - me.location
                dist = relative.magnitude()
                local = me.local(relative)

                # (NOVO) wavedash quando sem boost e longe, pra acelerar sem gastar boost
                if (me.boost < 12 and dist > 2800 and me.velocity.magnitude() < 1050 and
                    abs(local.x) < 220 and (agent.time - agent.wd_last) > 2.2 and
                    bool(getattr(me, "on_ground", True))):
                    agent.wd_last = agent.time
                    agent.dbg["action"] = "MOVE: WAVEDASH"
                    agent.push(WaveDash())
                else:
                    defaultPD(agent, local)

                    # (4) speed control: se estiver “chegando na bola” (perto), desacelera
                    desired_speed = _cap(dist * 2.0, 0, 2300)

                    # se muito perto da bola, ajusta timing
                    dist_ball = (ball.location - me.location).magnitude()
                    desired_speed = min(desired_speed, _desired_approach_speed(agent, dist_ball, ball.location.z, ball.velocity.z))

                    defaultThrottle(agent, desired_speed)
                    agent.controller.boost = (dist > 2600 and abs(local.x) < 240 and abs(local.y) > 900 and me.boost > 20)

        # ============================================================
        # RENDER AVANÇADO (pedido do usuário)
        # ============================================================

        # texto no carro
        _draw_text_3d(agent, me.location + Vector3(0, 0, 120),
                      f"[{agent.dbg['role']}] {agent.dbg['action']} | threat={threat:.2f}", (255, 255, 255))

        # se temos intercepto, desenha:
        it = agent.dbg["intercept_t"]
        il = agent.dbg["intercept_loc"]
        st = agent.dbg["shot_target"]

        if it is not None and il is not None:
            # 1) círculo no chão do intercepto
            ground = Vector3(il.x, il.y, 0)
            _draw_circle(agent, ground, radius=260, color=(100, 220, 255), steps=28, z_override=5)

            # 2) seta indicando direção do chute (alvo)
            if st is not None:
                arrow_end = Vector3(st.x, st.y, 0)
                # seta começa no círculo e aponta pro alvo (no chão)
                _draw_arrow(agent, ground + Vector3(0, 0, 10), arrow_end + Vector3(0, 0, 10), color=(255, 180, 80))

            # 3) path da bola até o intercepto (no ar)
            dt = max(0.0, it - agent.time)
            if dt > 0.05:
                pts = _sample_ball_path(ball.location, ball.velocity, min(dt, 4.0), step=0.10)

                # desenha polyline
                for i in range(len(pts) - 1):
                    _safe_line(agent, pts[i], pts[i + 1], (180, 255, 180))

                # marca ponto exato do toque (il)
                _draw_circle(agent, il, radius=120, color=(255, 255, 255), steps=18, z_override=il.z)

                # linha do carro até o intercepto (visual “vou pra lá”)
                _safe_line(agent, me.location, il, (255, 255, 0))
